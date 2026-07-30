from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .rel_att import (
    AUTHOR_QWEN_BBOX_SIZE,
    GENERAL_PROMPT,
    CropResult,
    _portable_source_path,
    _prepare_single_image_inputs,
    bbox_from_attention_adaptive,
)


@dataclass(frozen=True)
class GradAttentionResult:
    attention_map: np.ndarray
    attention: np.ndarray
    attention_gradient: np.ndarray
    grid_shape: tuple[int, int]
    image_token_start: int
    image_token_end: int
    layer: int
    target_token_id: int
    target_log_probability: float
    finite_difference: dict[str, Any] | None = None


@dataclass(frozen=True)
class PureGradientResult:
    attention_map: np.ndarray
    task_gradient_norm: np.ndarray
    general_gradient_norm: np.ndarray | None
    high_pass_mask: np.ndarray
    grid_shape: tuple[int, int]
    target_token_id: int
    target_log_probability: float
    normalize_by_general: bool
    median_kernel: int
    finite_difference: dict[str, Any] | None = None


def _prepared_forward_state(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
) -> dict[str, Any]:
    _, inputs = _prepare_single_image_inputs(
        model,
        processor,
        image_path,
        prompt,
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    attention_mask = inputs.get("attention_mask")
    model_kwargs = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "pixel_values", "attention_mask"}
    }
    token_ids = input_ids[0].tolist()
    try:
        image_start = token_ids.index(model.config.vision_start_token_id) + 1
        image_end = token_ids.index(model.config.vision_end_token_id)
    except ValueError as exc:
        raise ValueError("Vision token delimiters were not found in the prompt") from exc

    grid = np.asarray(inputs["image_grid_thw"][0].tolist(), dtype=np.int64)
    if int(grid[0]) != 1:
        raise ValueError(
            "Static-image gradient maps require image_grid_thw[0] == 1"
        )
    merge_size = int(model.config.vision_config.spatial_merge_size)
    grid_shape = (
        int(grid[1] // merge_size),
        int(grid[2] // merge_size),
    )
    expected_tokens = int(grid[0] * grid_shape[0] * grid_shape[1])
    if image_end - image_start != expected_tokens:
        raise ValueError(
            "Image token span and merged grid disagree: "
            f"{image_end - image_start} != {expected_tokens}"
        )
    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "model_kwargs": model_kwargs,
        "image_grid_thw": inputs["image_grid_thw"],
        "grid": grid,
        "grid_shape": grid_shape,
        "image_start": image_start,
        "image_end": image_end,
    }


def _forward_logits(
    model: Any,
    state: dict[str, Any],
    pixel_values: Any,
):
    embedding_output = model.get_input_embeddings(
        state["input_ids"],
        pixel_values,
        mask=state["attention_mask"],
        **state["model_kwargs"],
    )
    language_kwargs = {
        key: value
        for key, value in embedding_output.to_dict().items()
        if key != "inputs_embeds" and value is not None
    }
    return model.language_model(
        state["input_ids"],
        inputs_embeds=embedding_output.inputs_embeds,
        **language_kwargs,
    ).logits


def _target_log_probability(logits: Any, target_token_id: int):
    import mlx.core as mx

    final_logits = logits[0, -1].astype(mx.float32)
    return final_logits[target_token_id] - mx.logsumexp(final_logits)


def _select_target(model: Any, state: dict[str, Any]) -> tuple[int, float]:
    import mlx.core as mx

    logits = _forward_logits(model, state, state["pixel_values"])
    final_logits = logits[0, -1].astype(mx.float32)
    target_token_id = int(mx.argmax(final_logits).item())
    target_log_probability = _target_log_probability(
        logits,
        target_token_id,
    )
    mx.eval(target_log_probability)
    return target_token_id, float(target_log_probability.item())


class _DifferentiableAttention:
    """Explicit full attention with a differentiable final-row perturbation."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.final_row_delta = None
        self.last_attention = None

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def __call__(
        self,
        x,
        mask=None,
        cache=None,
        position_ids=None,
        position_embeddings=None,
    ):
        import mlx.core as mx

        if cache is not None:
            raise ValueError("Gradient-attention extraction must run without a KV cache")
        reference_output = self.inner(
            x,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        batch, length, _ = x.shape
        queries = self.inner.q_norm(
            self.inner.q_proj(x).reshape(
                batch,
                length,
                self.inner.n_heads,
                self.inner.head_dim,
            )
        ).transpose(0, 2, 1, 3)
        keys = self.inner.k_norm(
            self.inner.k_proj(x).reshape(
                batch,
                length,
                self.inner.n_kv_heads,
                self.inner.head_dim,
            )
        ).transpose(0, 2, 1, 3)
        values = self.inner.v_proj(x).reshape(
            batch,
            length,
            self.inner.n_kv_heads,
            self.inner.head_dim,
        ).transpose(0, 2, 1, 3)

        if position_embeddings is None:
            queries, keys = self.inner.rotary_emb.apply_rotary(
                queries,
                keys,
                position_ids,
                unsqueeze_dim=1,
            )
        else:
            from mlx_vlm.models.qwen3_vl.language import (
                apply_multimodal_rotary_pos_emb,
            )

            cos, sin = position_embeddings
            queries, keys = apply_multimodal_rotary_pos_emb(
                queries,
                keys,
                cos,
                sin,
            )

        repeats = self.inner.n_heads // self.inner.n_kv_heads
        if repeats > 1:
            keys = mx.repeat(keys, repeats=repeats, axis=1)
            values = mx.repeat(values, repeats=repeats, axis=1)

        scores = (
            queries.astype(mx.float32)
            @ keys.transpose(0, 1, 3, 2).astype(mx.float32)
        ) * self.inner.scale
        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError(f"Unsupported string attention mask: {mask}")
            query_indices = mx.arange(length)[:, None]
            key_indices = mx.arange(length)[None, :]
            allowed = query_indices >= key_indices
            scores = mx.where(
                allowed[None, None, :, :],
                scores,
                mx.array(-1e9, dtype=mx.float32),
            )
        elif isinstance(mask, mx.array):
            mask_values = mask[..., : scores.shape[-2], : scores.shape[-1]]
            if mask_values.dtype == mx.bool_:
                scores = mx.where(
                    mask_values,
                    scores,
                    mx.array(-1e9, dtype=mx.float32),
                )
            else:
                scores = scores + mask_values

        attention = mx.softmax(scores, axis=-1, precise=True)
        self.last_attention = attention
        if self.final_row_delta is None:
            return reference_output

        if self.final_row_delta is not None:
            expected = (batch, self.inner.n_heads, 1, length)
            if self.final_row_delta.shape != expected:
                raise ValueError(
                    "Final-row perturbation has wrong shape: "
                    f"{self.final_row_delta.shape} != {expected}"
                )
            perturbed_attention = mx.concatenate(
                [
                    attention[:, :, :-1, :],
                    attention[:, :, -1:, :] + self.final_row_delta,
                ],
                axis=2,
            )

        base_attended = (
            attention.astype(mx.float32) @ values.astype(mx.float32)
        ).transpose(0, 2, 1, 3).reshape(batch, length, -1)
        perturbed_attended = (
            perturbed_attention.astype(mx.float32) @ values.astype(mx.float32)
        ).transpose(0, 2, 1, 3).reshape(batch, length, -1)
        base_projected = self.inner.o_proj(base_attended.astype(x.dtype))
        perturbed_projected = self.inner.o_proj(
            perturbed_attended.astype(x.dtype)
        )
        # Keep MLX-VLM's fused SDPA forward value exactly, while adding the
        # explicit attention operator's change under a differentiable
        # perturbation.  At delta=0 the two explicit projections cancel; the
        # derivative with respect to delta is the required d v / d A.
        return reference_output + (perturbed_projected - base_projected)


def _finite_difference_diagnostic(
    objective: Any,
    gradient: Any,
    shape: tuple[int, ...],
    *,
    epsilons: tuple[float, ...],
) -> dict[str, Any]:
    import mlx.core as mx

    gradient_np = np.asarray(
        gradient.astype(mx.float32).tolist(),
        dtype=np.float32,
    )
    flat_index = int(np.argmax(np.abs(gradient_np)))
    coordinate = tuple(int(value) for value in np.unravel_index(flat_index, shape))
    direction_np = np.zeros(shape, dtype=np.float32)
    direction_np[coordinate] = 1.0
    direction = mx.array(direction_np)
    analytic = mx.sum(gradient.astype(mx.float32) * direction)
    forward = None
    forward_error = None
    try:
        _, forward_products = mx.jvp(
            objective,
            [mx.zeros(shape, dtype=mx.float32)],
            [direction],
        )
        forward = forward_products[0]
        mx.eval(analytic, forward)
    except ValueError as exc:
        # MLX 0.32.0 has no forward-mode rule for the quantized CustomKernel
        # used by this model. Reverse-mode gradients remain available.
        forward_error = str(exc)
        mx.eval(analytic)
    analytic_value = float(analytic.item())
    forward_value = float(forward.item()) if forward is not None else None
    autodiff_absolute_error = (
        abs(analytic_value - forward_value)
        if forward_value is not None
        else None
    )
    autodiff_relative_error = (
        autodiff_absolute_error
        / max(abs(analytic_value), abs(forward_value), 1e-12)
        if forward_value is not None
        else None
    )
    measurements = []
    for epsilon in epsilons:
        plus = objective(direction * epsilon)
        mx.eval(plus)
        plus_value = float(plus.item())
        minus = objective(direction * -epsilon)
        mx.eval(minus)
        minus_value = float(minus.item())
        finite_value = (plus_value - minus_value) / (2.0 * epsilon)
        absolute_error = abs(analytic_value - finite_value)
        relative_error = absolute_error / max(
            abs(analytic_value),
            abs(finite_value),
            1e-12,
        )
        measurements.append(
            {
                "epsilon": epsilon,
                "finite_difference_directional_derivative": finite_value,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
            }
        )
    best = min(measurements, key=lambda item: item["relative_error"])
    return {
        "direction": "one-hot coordinate with maximum absolute analytic gradient",
        "coordinate": list(coordinate),
        "analytic_directional_derivative": analytic_value,
        "forward_mode_directional_derivative": forward_value,
        "reverse_vs_forward_absolute_error": autodiff_absolute_error,
        "reverse_vs_forward_relative_error": autodiff_relative_error,
        "forward_mode_unavailable": forward_error,
        "measurements": measurements,
        "best_relative_error": best["relative_error"],
        "best_epsilon": best["epsilon"],
    }


def gradient_weighted_attention_qwen3_vl(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    *,
    layer: int,
    validate_gradient: bool = False,
) -> GradAttentionResult:
    """Paper grad-att for Qwen3-VL using MLX automatic differentiation."""

    import mlx.core as mx

    state = _prepared_forward_state(model, processor, image_path, prompt)
    target_token_id, target_log_probability = _select_target(model, state)

    embedding_output = model.get_input_embeddings(
        state["input_ids"],
        state["pixel_values"],
        mask=state["attention_mask"],
        **state["model_kwargs"],
    )
    language_kwargs = {
        key: value
        for key, value in embedding_output.to_dict().items()
        if key != "inputs_embeds" and value is not None
    }

    layers = model.language_model.model.layers
    if layer < 0 or layer >= len(layers):
        raise IndexError(f"Attention layer {layer} is outside 0..{len(layers) - 1}")
    original = layers[layer].self_attn
    wrapper = _DifferentiableAttention(original)
    layers[layer].self_attn = wrapper
    length = int(state["input_ids"].shape[-1])
    delta_shape = (1, original.n_heads, 1, length)
    zero_delta = mx.zeros(delta_shape, dtype=mx.float32)

    def objective(delta):
        wrapper.final_row_delta = delta
        logits = model.language_model(
            state["input_ids"],
            inputs_embeds=embedding_output.inputs_embeds,
            **language_kwargs,
        ).logits
        return _target_log_probability(logits, target_token_id)

    try:
        gradient = mx.grad(objective)(zero_delta)
        if wrapper.last_attention is None:
            raise RuntimeError("Differentiable attention layer did not run")
        attention = wrapper.last_attention
        mx.eval(gradient, attention)
        finite_difference = (
            _finite_difference_diagnostic(
                objective,
                gradient,
                delta_shape,
                epsilons=(0.1, 0.03, 0.01, 0.003, 0.001),
            )
            if validate_gradient
            else None
        )
    finally:
        layers[layer].self_attn = original
        mx.clear_cache()

    image_slice = slice(state["image_start"], state["image_end"])
    attention_np = np.asarray(
        attention[0, :, -1, image_slice].astype(mx.float32).tolist(),
        dtype=np.float64,
    )
    gradient_np = np.asarray(
        gradient[0, :, 0, image_slice].astype(mx.float32).tolist(),
        dtype=np.float64,
    )
    weighted = attention_np * np.maximum(gradient_np, 0.0)
    attention_map = weighted.mean(axis=0).reshape(state["grid_shape"])
    if not np.isfinite(attention_map).all():
        raise FloatingPointError("Gradient-weighted attention map is non-finite")
    if not np.any(attention_map > 0):
        raise FloatingPointError("Gradient-weighted attention map has no positive value")

    return GradAttentionResult(
        attention_map=attention_map,
        attention=attention_np,
        attention_gradient=gradient_np,
        grid_shape=state["grid_shape"],
        image_token_start=state["image_start"],
        image_token_end=state["image_end"],
        layer=layer,
        target_token_id=target_token_id,
        target_log_probability=target_log_probability,
        finite_difference=finite_difference,
    )


def unpatchify_qwen3_vl_pixel_gradient(
    flattened_gradient: np.ndarray,
    grid_thw: tuple[int, int, int],
    *,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    channels: int = 3,
) -> np.ndarray:
    """Reverse Qwen3-VL processor patch ordering and sum duplicate frames."""

    grid_t, grid_h, grid_w = map(int, grid_thw)
    expected = (
        grid_t * grid_h * grid_w,
        channels * temporal_patch_size * patch_size * patch_size,
    )
    if flattened_gradient.shape != expected:
        raise ValueError(
            f"Flattened pixel gradient shape {flattened_gradient.shape} != {expected}"
        )
    transposed = flattened_gradient.reshape(
        1,
        grid_t,
        grid_h // merge_size,
        grid_w // merge_size,
        merge_size,
        merge_size,
        channels,
        temporal_patch_size,
        patch_size,
        patch_size,
    )
    patches = transposed.transpose(0, 1, 7, 6, 2, 4, 8, 3, 5, 9)
    frames = patches.reshape(
        grid_t,
        temporal_patch_size,
        channels,
        grid_h * patch_size,
        grid_w * patch_size,
    )
    return frames.sum(axis=(0, 1))


def _high_pass_mask(
    image_path: Path,
    resized_shape: tuple[int, int],
    *,
    median_kernel: int,
) -> np.ndarray:
    if median_kernel < 3 or median_kernel % 2 == 0:
        raise ValueError("median_kernel must be an odd integer of at least 3")
    resized_h, resized_w = resized_shape
    with Image.open(image_path) as source:
        rgb = source.convert("RGB").resize(
            (resized_w, resized_h),
            Image.Resampling.BICUBIC,
        )
        image = np.asarray(rgb, dtype=np.float32) / 255.0
    blurred = cv2.GaussianBlur(
        image,
        (3, 3),
        sigmaX=0.8,
        sigmaY=0.8,
        borderType=cv2.BORDER_REFLECT_101,
    )
    high_frequency = image - blurred
    brightness = np.sqrt(np.square(high_frequency).sum(axis=2))
    filtered = cv2.medianBlur(brightness.astype(np.float32), median_kernel)
    return filtered > np.median(filtered)


def _pool_to_merged_grid(
    values: np.ndarray,
    grid_thw: tuple[int, int, int],
    *,
    patch_size: int,
    merge_size: int,
) -> np.ndarray:
    grid_t, grid_h, grid_w = map(int, grid_thw)
    if grid_t != 1:
        raise ValueError("Pure-gradient image pooling currently requires grid_t=1")
    block = patch_size * merge_size
    expected_shape = (grid_h * patch_size, grid_w * patch_size)
    if values.shape != expected_shape:
        raise ValueError(f"Pixel map shape {values.shape} != {expected_shape}")
    return values.reshape(
        grid_h // merge_size,
        block,
        grid_w // merge_size,
        block,
    ).mean(axis=(1, 3))


def _pixel_gradient(
    model: Any,
    state: dict[str, Any],
    *,
    validate_gradient: bool,
) -> tuple[np.ndarray, int, float, dict[str, Any] | None]:
    import mlx.core as mx

    target_token_id, target_log_probability = _select_target(model, state)

    pixel_values = state["pixel_values"]
    vision_dtype = model.vision_tower.patch_embed.proj.weight.dtype

    vision_tower = model.vision_tower
    grid_thw = state["image_grid_thw"]

    def vision_outputs_from_patch_embeddings(patch_embeddings):
        hidden_states = patch_embeddings + vision_tower.fast_pos_embed_interpolate(
            grid_thw
        )
        rotary_pos_emb = vision_tower.rot_pos_emb(grid_thw)
        sequence_length = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(sequence_length, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(sequence_length, -1)

        cumulative_lengths = []
        for index in range(grid_thw.shape[0]):
            sequence = grid_thw[index, 1] * grid_thw[index, 2]
            cumulative_lengths.append(mx.repeat(sequence, grid_thw[index, 0]))
        cumulative_lengths = mx.concatenate(cumulative_lengths)
        cumulative_lengths = mx.cumsum(
            cumulative_lengths.astype(mx.int32),
            axis=0,
        )
        cumulative_lengths = mx.pad(
            cumulative_lengths,
            (1, 0),
            mode="constant",
            constant_values=0,
        )

        deepstack_features = []
        for layer_number, block in enumerate(vision_tower.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cumulative_lengths,
                rotary_pos_emb=rotary_pos_emb,
            )
            if layer_number in vision_tower.deepstack_visual_indexes:
                merger_index = vision_tower.deepstack_visual_indexes.index(
                    layer_number
                )
                deepstack_features.append(
                    vision_tower.deepstack_merger_list[merger_index](hidden_states)
                )
        main_features = vision_tower.merger(hidden_states)
        return [main_features, *deepstack_features]

    patch_embeddings = vision_tower.patch_embed(pixel_values.astype(vision_dtype))
    mx.eval(patch_embeddings)
    detached_patch_embeddings = mx.stop_gradient(patch_embeddings)

    input_ids = state["input_ids"]
    text_embeddings = model.language_model.model.embed_tokens(input_ids)
    position_ids, rope_deltas = model.language_model.get_rope_index(
        input_ids,
        state["image_grid_thw"],
        None,
        state["attention_mask"],
    )
    visual_pos_masks = input_ids == model.config.image_token_index
    image_start = state["image_start"]
    image_end = state["image_end"]

    def language_objective(main_features, *deepstack_features):
        if main_features.shape[0] != image_end - image_start:
            raise ValueError(
                "Vision feature count does not match the image-token span"
            )
        inputs_embeds = mx.concatenate(
            [
                text_embeddings[:, :image_start, :],
                main_features[None, :, :],
                text_embeddings[:, image_end:, :],
            ],
            axis=1,
        )
        logits = model.language_model(
            input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=list(deepstack_features),
        ).logits
        return _target_log_probability(logits, target_token_id)

    patch_embed = vision_tower.patch_embed
    weight = patch_embed.proj.weight.astype(mx.float32)
    bias = patch_embed.proj.bias.astype(mx.float32)

    def explicit_patch_embed(flat_pixels):
        convolution_order = (
            flat_pixels.astype(mx.float32)
            .reshape(
                -1,
                patch_embed.in_channels,
                patch_embed.temporal_patch_size,
                patch_embed.patch_size,
                patch_embed.patch_size,
            )
            .moveaxis(1, 4)
            .reshape(flat_pixels.shape[0], -1)
        )
        return (
            convolution_order @ weight.reshape(weight.shape[0], -1).T + bias
        ).astype(vision_dtype)

    explicit_patch_anchor = mx.stop_gradient(
        explicit_patch_embed(pixel_values)
    )
    mx.eval(explicit_patch_anchor)

    def objective(flat_pixels):
        # MLX 0.32's Conv3d VJP attempts an impractically large allocation for
        # Qwen's kernel=stride patch embedding.  The convolution is exactly one
        # affine transform per non-overlapping flattened patch, so replace only
        # that operator by the equivalent matrix multiplication.  Anchoring the
        # value to the native Conv3d output preserves the exact forward point;
        # automatic differentiation then traverses the whole vision+language
        # graph in one reverse pass.
        patch_delta = explicit_patch_embed(flat_pixels) - explicit_patch_anchor
        anchored_patches = detached_patch_embeddings + patch_delta
        return language_objective(
            *vision_outputs_from_patch_embeddings(anchored_patches)
        )

    gradient = mx.grad(objective)(pixel_values)
    mx.eval(gradient)
    finite_difference = (
        _finite_difference_diagnostic(
            objective,
            gradient,
            tuple(int(size) for size in gradient.shape),
            epsilons=(0.03, 0.01, 0.003, 0.001),
        )
        if validate_gradient
        else None
    )
    gradient_np = np.asarray(
        gradient.astype(mx.float32).tolist(),
        dtype=np.float32,
    )
    mx.clear_cache()
    return (
        gradient_np,
        target_token_id,
        target_log_probability,
        finite_difference,
    )


def input_gradient_qwen3_vl(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    *,
    normalize_by_general: bool = False,
    general_prompt: str = GENERAL_PROMPT,
    median_kernel: int = 3,
    validate_gradient: bool = False,
) -> PureGradientResult:
    """Paper pure-grad, with optional public-repository normalization.

    The paper defines ||d v / d x|| followed by an edge mask.  The public
    LLaVA/BLIP code additionally divides by the general-prompt gradient and
    uses a 7x7 median filter.  ``normalize_by_general=True, median_kernel=7``
    reproduces that code path while the defaults reproduce the paper text.
    """

    state = _prepared_forward_state(model, processor, image_path, prompt)
    (
        task_flat_gradient,
        target_token_id,
        target_log_probability,
        finite_difference,
    ) = _pixel_gradient(
        model,
        state,
        validate_gradient=validate_gradient,
    )

    config = model.config.vision_config
    patch_size = int(config.patch_size)
    temporal_patch_size = int(config.temporal_patch_size)
    merge_size = int(config.spatial_merge_size)
    grid = tuple(int(value) for value in state["grid"])
    task_gradient = unpatchify_qwen3_vl_pixel_gradient(
        task_flat_gradient,
        grid,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        merge_size=merge_size,
    )
    task_norm = np.linalg.norm(task_gradient.astype(np.float64), axis=0)

    general_norm = None
    importance = task_norm
    if normalize_by_general:
        general_state = _prepared_forward_state(
            model,
            processor,
            image_path,
            general_prompt,
        )
        if tuple(general_state["grid_shape"]) != tuple(state["grid_shape"]):
            raise ValueError("Task and general prompts produced different visual grids")
        general_flat_gradient, _, _, _ = _pixel_gradient(
            model,
            general_state,
            validate_gradient=False,
        )
        general_gradient = unpatchify_qwen3_vl_pixel_gradient(
            general_flat_gradient,
            grid,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
        )
        general_norm = np.linalg.norm(general_gradient.astype(np.float64), axis=0)
        if np.any(general_norm <= 0):
            raise FloatingPointError(
                "General-prompt pixel-gradient norm contains zero"
            )
        importance = task_norm / general_norm

    resized_shape = (
        int(grid[1] * patch_size),
        int(grid[2] * patch_size),
    )
    high_pass_mask = _high_pass_mask(
        image_path,
        resized_shape,
        median_kernel=median_kernel,
    )
    importance = importance * high_pass_mask
    attention_map = _pool_to_merged_grid(
        importance,
        grid,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    if not np.isfinite(attention_map).all():
        raise FloatingPointError("Pure-gradient map is non-finite")
    if not np.any(attention_map > 0):
        raise FloatingPointError("Pure-gradient map has no positive value")

    return PureGradientResult(
        attention_map=attention_map,
        task_gradient_norm=task_norm,
        general_gradient_norm=general_norm,
        high_pass_mask=high_pass_mask,
        grid_shape=state["grid_shape"],
        target_token_id=target_token_id,
        target_log_probability=target_log_probability,
        normalize_by_general=normalize_by_general,
        median_kernel=median_kernel,
        finite_difference=finite_difference,
    )


def save_gradient_map_crop(
    image_path: Path,
    output_dir: Path,
    result: GradAttentionResult | PureGradientResult,
    *,
    method: str,
    bbox_size: int = AUTHOR_QWEN_BBOX_SIZE,
) -> tuple[Path, CropResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        crop_result = bbox_from_attention_adaptive(
            result.attention_map,
            image.size,
            bbox_size=bbox_size,
        )
        crop_path = output_dir / "crop.png"
        image.crop(crop_result.bbox).save(crop_path)

    np.save(output_dir / "importance_map.npy", result.attention_map)
    if isinstance(result, GradAttentionResult):
        np.save(output_dir / "attention.npy", result.attention)
        np.save(output_dir / "attention_gradient.npy", result.attention_gradient)
        details = {
            "attention_layer": result.layer,
            "image_token_span": [
                result.image_token_start,
                result.image_token_end,
            ],
        }
    else:
        np.savez_compressed(
            output_dir / "pixel_gradient_maps.npz",
            task_gradient_norm=result.task_gradient_norm.astype(np.float32),
            high_pass_mask=result.high_pass_mask,
            **(
                {
                    "general_gradient_norm": (
                        result.general_gradient_norm.astype(np.float32)
                    )
                }
                if result.general_gradient_norm is not None
                else {}
            ),
        )
        details = {
            "normalize_by_general": result.normalize_by_general,
            "median_kernel": result.median_kernel,
        }

    metadata = {
        "source_image": _portable_source_path(image_path),
        "method": method,
        "grid_shape": list(result.grid_shape),
        "target_token_id": result.target_token_id,
        "target_log_probability": result.target_log_probability,
        "finite_difference": result.finite_difference,
        "bbox_size": bbox_size,
        "crop": asdict(crop_result),
        **details,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return crop_path, crop_result
