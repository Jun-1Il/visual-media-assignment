from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


GENERAL_PROMPT = (
    "Write a general description of the image. "
    "Answer the question using a single word or phrase."
)
AUTHOR_QWEN_ATTENTION_LAYER = 22
AUTHOR_QWEN_BBOX_SIZE = 224
AUTHOR_CROP_RATIOS = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)


def _portable_source_path(image_path: Path) -> str:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        return image_path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return image_path.as_posix()


@dataclass(frozen=True)
class RelativeAttentionResult:
    attention_map: np.ndarray
    task_attention: np.ndarray
    general_attention: np.ndarray
    grid_shape: tuple[int, int]
    image_token_start: int
    image_token_end: int
    layer: int


@dataclass(frozen=True)
class CropResult:
    bbox: tuple[int, int, int, int]
    ratio: float
    window_blocks: tuple[int, int]
    attention_difference: float


class _AttentionCapture:
    """Wrap one MLX Qwen3-VL attention layer and expose its actual softmax map.

    MLX-VLM uses the fused SDPA kernel, whose public return value contains only
    the attended values.  This wrapper reuses the loaded layer's quantized
    q/k projections, q/k normalisation, and mRoPE inputs, then evaluates only
    the final query row.  The original layer still produces the model output.
    """

    def __init__(self, inner: Any, *, validate_reconstruction: bool = False):
        self.inner = inner
        self.validate_reconstruction = validate_reconstruction
        self.last_attention = None
        self.last_reconstruction_diagnostics = None

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

        output = self.inner(
            x,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        if cache is not None:
            raise ValueError("Relative-attention capture must run without a KV cache")

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

        scores = (
            queries[:, :, -1:, :].astype(mx.float32)
            @ keys.transpose(0, 1, 3, 2).astype(mx.float32)
        ) * self.inner.scale

        # In prefill MLX represents an ordinary causal mask as the string
        # "causal".  Its final row admits every key, so no addition is needed.
        if isinstance(mask, mx.array):
            mask_row = mask
            if mask_row.ndim >= 2:
                mask_row = mask_row[..., -1:, : scores.shape[-1]]
            if mask_row.dtype == mx.bool_:
                mask_row = mx.where(
                    mask_row,
                    mx.array(0.0, dtype=mx.float32),
                    mx.array(-1e9, dtype=mx.float32),
                )
            scores = scores + mask_row

        attention = mx.softmax(scores, axis=-1, precise=True)
        self.last_attention = attention.mean(axis=1)

        if self.validate_reconstruction:
            values = self.inner.v_proj(x).reshape(
                batch,
                length,
                self.inner.n_kv_heads,
                self.inner.head_dim,
            ).transpose(0, 2, 1, 3)
            if repeats > 1:
                values = mx.repeat(values, repeats=repeats, axis=1)

            attended = (
                attention.astype(mx.float32) @ values.astype(mx.float32)
            ).transpose(0, 2, 1, 3).reshape(batch, 1, -1)
            reconstructed = self.inner.o_proj(attended.astype(output.dtype))
            reference = output[:, -1:, :].astype(mx.float32)
            difference = reconstructed.astype(mx.float32) - reference
            reference_l2 = mx.sqrt(mx.sum(mx.square(reference)))
            difference_l2 = mx.sqrt(mx.sum(mx.square(difference)))
            reference_max = mx.max(mx.abs(reference))
            self.last_reconstruction_diagnostics = {
                "max_abs_error": mx.max(mx.abs(difference)),
                "mean_abs_error": mx.mean(mx.abs(difference)),
                "relative_l2_error": difference_l2
                / mx.maximum(reference_l2, mx.array(1e-12, dtype=mx.float32)),
                "relative_max_error": mx.max(mx.abs(difference))
                / mx.maximum(reference_max, mx.array(1e-12, dtype=mx.float32)),
                "attention_row_sum_error": mx.max(
                    mx.abs(attention.sum(axis=-1) - 1.0)
                ),
                "reference_max_abs": reference_max,
            }

        arrays_to_evaluate = [self.last_attention]
        if self.last_reconstruction_diagnostics is not None:
            arrays_to_evaluate.extend(self.last_reconstruction_diagnostics.values())
        mx.eval(*arrays_to_evaluate)
        return output


def configure_author_qwen_resolution(processor: Any, model: Any) -> int:
    """Match the official Qwen ViCrop budget of 256 merged visual tokens."""

    patch_size = int(model.config.vision_config.patch_size)
    merge_size = int(model.config.vision_config.spatial_merge_size)
    max_pixels = 256 * (patch_size * merge_size) ** 2
    image_processor = processor.image_processor
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels
    elif hasattr(image_processor, "size") and hasattr(
        image_processor.size,
        "__setitem__",
    ):
        image_processor.size["longest_edge"] = max_pixels
    else:
        raise TypeError(
            "Qwen image processor must expose max_pixels or a mutable size mapping"
        )
    return max_pixels


def _prepare_single_image_inputs(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
):
    from mlx_vlm import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    formatted_prompt = apply_chat_template(
        processor,
        model.config,
        prompt,
        num_images=1,
    )
    inputs = prepare_inputs(
        processor,
        images=[str(image_path)],
        prompts=formatted_prompt,
        image_token_index=model.config.image_token_index,
        add_special_tokens=True,
    )
    return formatted_prompt, inputs


def _capture_prompt_attentions(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    layers_to_capture: tuple[int, ...],
    reconstruction_diagnostics: dict[int, dict[str, float]] | None = None,
) -> tuple[dict[int, np.ndarray], np.ndarray, tuple[int, int], int, int]:
    import mlx.core as mx

    _, inputs = _prepare_single_image_inputs(
        model,
        processor,
        image_path,
        prompt,
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    attention_mask = inputs.get("attention_mask")
    image_grid_thw = inputs["image_grid_thw"]
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

    merge_size = int(model.config.vision_config.spatial_merge_size)
    grid = np.asarray(image_grid_thw[0].tolist(), dtype=np.int64)
    if int(grid[0]) != 1:
        raise ValueError("Static-image rel-att requires image_grid_thw[0] == 1")
    grid_h = int(grid[1] // merge_size)
    grid_w = int(grid[2] // merge_size)
    expected_tokens = int(grid[0] * grid_h * grid_w)
    if image_end - image_start != expected_tokens:
        raise ValueError(
            "Image token span and merged grid disagree: "
            f"{image_end - image_start} != {expected_tokens}"
        )

    layers = model.language_model.model.layers
    if not layers_to_capture:
        raise ValueError("At least one attention layer must be requested")
    if len(set(layers_to_capture)) != len(layers_to_capture):
        raise ValueError("Attention layers must be unique")
    for layer in layers_to_capture:
        if layer < 0 or layer >= len(layers):
            raise IndexError(
                f"Attention layer {layer} is outside 0..{len(layers) - 1}"
            )

    originals = {
        layer: layers[layer].self_attn for layer in layers_to_capture
    }
    captures = {
        layer: _AttentionCapture(
            originals[layer],
            validate_reconstruction=reconstruction_diagnostics is not None,
        )
        for layer in layers_to_capture
    }
    for layer, capture in captures.items():
        layers[layer].self_attn = capture
    try:
        embedding_output = model.get_input_embeddings(
            input_ids,
            pixel_values,
            mask=attention_mask,
            **model_kwargs,
        )
        language_kwargs = {
            key: value
            for key, value in embedding_output.to_dict().items()
            if key != "inputs_embeds" and value is not None
        }
        outputs = model.language_model(
            input_ids,
            inputs_embeds=embedding_output.inputs_embeds,
            **language_kwargs,
        )
        captured_arrays = [captures[layer].last_attention for layer in layers_to_capture]
        if any(array is None for array in captured_arrays):
            raise RuntimeError("At least one selected attention layer did not run")
        mx.eval(outputs.logits, *captured_arrays)
        full_attentions = {
            layer: np.asarray(captures[layer].last_attention[0, 0].tolist())
            for layer in layers_to_capture
        }
        if reconstruction_diagnostics is not None:
            for layer, capture in captures.items():
                if capture.last_reconstruction_diagnostics is None:
                    raise RuntimeError(
                        f"Layer {layer} did not produce reconstruction diagnostics"
                    )
                reconstruction_diagnostics[layer] = {
                    name: float(value.item())
                    for name, value in (
                        capture.last_reconstruction_diagnostics.items()
                    )
                }
    finally:
        for layer, original in originals.items():
            layers[layer].self_attn = original
        mx.clear_cache()

    image_attentions = {
        layer: attention[image_start:image_end].astype(
            np.float64,
            copy=False,
        )
        for layer, attention in full_attentions.items()
    }
    return image_attentions, grid, (grid_h, grid_w), image_start, image_end


def validate_attention_reconstruction_qwen3_vl(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    *,
    layer: int,
) -> dict[str, Any]:
    """Numerically compare explicit attention with MLX fused-SDPA output.

    The final-query row is reconstructed as softmax(QK^T / sqrt(d)) @ V,
    followed by the loaded output projection.  Agreement with the original
    fused layer output validates Q/K projection, QK normalisation, mRoPE,
    grouped-query head expansion, scaling, masking, and softmax together.
    """

    diagnostics: dict[int, dict[str, float]] = {}
    _, _, grid_shape, image_start, image_end = _capture_prompt_attentions(
        model,
        processor,
        image_path,
        prompt,
        (layer,),
        reconstruction_diagnostics=diagnostics,
    )
    return {
        "layer": layer,
        "grid_shape": list(grid_shape),
        "image_token_span": [image_start, image_end],
        **diagnostics[layer],
    }


def _capture_prompt_attention(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], int, int]:
    attentions, grid, grid_shape, image_start, image_end = (
        _capture_prompt_attentions(
            model,
            processor,
            image_path,
            prompt,
            (layer,),
        )
    )
    return attentions[layer], grid, grid_shape, image_start, image_end


def relative_attention_qwen3_vl(
    model: Any,
    processor: Any,
    image_path: Path,
    question: str,
    *,
    general_prompt: str = GENERAL_PROMPT,
    layer: int = AUTHOR_QWEN_ATTENTION_LAYER,
) -> RelativeAttentionResult:
    """Port the authors' Qwen2.5-VL rel-att equation to MLX Qwen3-VL."""

    task_attention, _, grid_shape, image_start, image_end = (
        _capture_prompt_attention(
            model,
            processor,
            image_path,
            question,
            layer,
        )
    )
    general_attention, _, general_grid_shape, general_start, general_end = (
        _capture_prompt_attention(
            model,
            processor,
            image_path,
            general_prompt,
            layer,
        )
    )
    if (grid_shape, image_start, image_end) != (
        general_grid_shape,
        general_start,
        general_end,
    ):
        raise ValueError("Task and general prompts produced different image layouts")
    if np.any(general_attention <= 0):
        raise FloatingPointError(
            "General-prompt attention contains zero; exact division is undefined"
        )

    attention_map = (task_attention / general_attention).reshape(grid_shape)
    if not np.isfinite(attention_map).all():
        raise FloatingPointError("Relative-attention map contains NaN or infinity")
    return RelativeAttentionResult(
        attention_map=attention_map,
        task_attention=task_attention,
        general_attention=general_attention,
        grid_shape=grid_shape,
        image_token_start=image_start,
        image_token_end=image_end,
        layer=layer,
    )


def relative_attention_layers_qwen3_vl(
    model: Any,
    processor: Any,
    image_path: Path,
    question: str,
    layers: tuple[int, ...],
    *,
    general_prompt: str = GENERAL_PROMPT,
) -> dict[int, RelativeAttentionResult]:
    """Extract rel-att maps for several layers using two shared model forwards."""

    task_attentions, _, grid_shape, image_start, image_end = (
        _capture_prompt_attentions(
            model,
            processor,
            image_path,
            question,
            layers,
        )
    )
    general_attentions, _, general_grid_shape, general_start, general_end = (
        _capture_prompt_attentions(
            model,
            processor,
            image_path,
            general_prompt,
            layers,
        )
    )
    if (grid_shape, image_start, image_end) != (
        general_grid_shape,
        general_start,
        general_end,
    ):
        raise ValueError("Task and general prompts produced different image layouts")

    results = {}
    for layer in layers:
        task_attention = task_attentions[layer]
        general_attention = general_attentions[layer]
        if np.any(general_attention <= 0):
            raise FloatingPointError(
                f"Layer {layer} general-prompt attention contains zero"
            )
        attention_map = (task_attention / general_attention).reshape(grid_shape)
        if not np.isfinite(attention_map).all():
            raise FloatingPointError(
                f"Layer {layer} relative-attention map is non-finite"
            )
        results[layer] = RelativeAttentionResult(
            attention_map=attention_map,
            task_attention=task_attention,
            general_attention=general_attention,
            grid_shape=grid_shape,
            image_token_start=image_start,
            image_token_end=image_end,
            layer=layer,
        )
    return results


def ensemble_log_relative_attention(
    results: dict[int, RelativeAttentionResult],
    selected_layers: tuple[int, ...],
) -> np.ndarray:
    """Robust improvement: average per-layer standardized log attention ratios."""

    maps = []
    for layer in selected_layers:
        result = results[layer]
        log_ratio = np.log(result.task_attention) - np.log(
            result.general_attention
        )
        median = float(np.median(log_ratio))
        mad = float(np.median(np.abs(log_ratio - median)))
        scale = max(1.4826 * mad, 1e-6)
        maps.append(((log_ratio - median) / scale).reshape(result.grid_shape))
    return np.mean(np.stack(maps, axis=0), axis=0)


def bbox_from_attention_adaptive(
    attention_map: np.ndarray,
    image_size: tuple[int, int],
    *,
    bbox_size: int = AUTHOR_QWEN_BBOX_SIZE,
    ratios: tuple[float, ...] = AUTHOR_CROP_RATIOS,
) -> CropResult:
    """Faithful NumPy port of the official adaptive attention-window search."""

    if attention_map.ndim != 2 or min(attention_map.shape) < 1:
        raise ValueError("attention_map must be a non-empty 2D array")
    image_width, image_height = image_size
    block_size = (
        image_width / attention_map.shape[1],
        image_height / attention_map.shape[0],
    )
    candidates: list[tuple[float, tuple[int, int], tuple[int, int], float]] = []

    for ratio in ratios:
        blocks_x = min(
            int(bbox_size * ratio / block_size[0]),
            attention_map.shape[1],
        )
        blocks_y = min(
            int(bbox_size * ratio / block_size[1]),
            attention_map.shape[0],
        )
        blocks_x = max(1, blocks_x)
        blocks_y = max(1, blocks_y)
        if (
            attention_map.shape[1] - blocks_x < 1
            and attention_map.shape[0] - blocks_y < 1
        ):
            if ratio == ratios[0]:
                return CropResult(
                    bbox=(0, 0, image_width, image_height),
                    ratio=float(ratio),
                    window_blocks=(blocks_x, blocks_y),
                    attention_difference=0.0,
                )
            continue

        sliding = np.empty(
            (
                attention_map.shape[0] - blocks_y + 1,
                attention_map.shape[1] - blocks_x + 1,
            ),
            dtype=np.float64,
        )
        for y in range(sliding.shape[0]):
            for x in range(sliding.shape[1]):
                sliding[y, x] = attention_map[
                    y : y + blocks_y,
                    x : x + blocks_x,
                ].sum()

        max_y, max_x = np.unravel_index(np.argmax(sliding), sliding.shape)
        max_attention = float(sliding[max_y, max_x])
        adjacent = []
        if max_x > 0:
            adjacent.append(sliding[max_y, max_x - 1])
        if max_x < sliding.shape[1] - 1:
            adjacent.append(sliding[max_y, max_x + 1])
        if max_y > 0:
            adjacent.append(sliding[max_y - 1, max_x])
        if max_y < sliding.shape[0] - 1:
            adjacent.append(sliding[max_y + 1, max_x])
        if not adjacent:
            difference = 0.0
        else:
            difference = (
                max_attention - float(np.mean(adjacent))
            ) / (blocks_x * blocks_y)
        candidates.append(
            (
                difference,
                (int(max_x), int(max_y)),
                (blocks_x, blocks_y),
                float(ratio),
            )
        )

    if not candidates:
        return CropResult(
            bbox=(0, 0, image_width, image_height),
            ratio=1.0,
            window_blocks=(attention_map.shape[1], attention_map.shape[0]),
            attention_difference=0.0,
        )

    difference, max_position, window_blocks, selected_ratio = max(
        candidates,
        key=lambda candidate: candidate[0],
    )
    selected_size = bbox_size * selected_ratio
    center_x = int(
        max_position[0] * block_size[0]
        + block_size[0] * window_blocks[0] / 2
    )
    center_y = int(
        max_position[1] * block_size[1]
        + block_size[1] * window_blocks[1] / 2
    )
    half = selected_size // 2
    center_x = max(half, min(center_x, image_width - half))
    center_y = max(half, min(center_y, image_height - half))
    bbox = (
        int(max(0, center_x - half)),
        int(max(0, center_y - half)),
        int(min(image_width, center_x + half)),
        int(min(image_height, center_y + half)),
    )
    return CropResult(
        bbox=bbox,
        ratio=selected_ratio,
        window_blocks=window_blocks,
        attention_difference=float(difference),
    )


def topk_bboxes_from_attention(
    attention_map: np.ndarray,
    image_size: tuple[int, int],
    *,
    k: int = 2,
    bbox_size: int = AUTHOR_QWEN_BBOX_SIZE,
) -> list[CropResult]:
    """Select multiple regions by suppressing grid cells covered by earlier crops."""

    if k < 1:
        raise ValueError("k must be at least one")
    working = np.asarray(attention_map, dtype=np.float64).copy()
    results: list[CropResult] = []
    image_width, image_height = image_size
    cell_width = image_width / working.shape[1]
    cell_height = image_height / working.shape[0]
    suppression_value = float(working.min() - max(1.0, np.ptp(working)))

    for _ in range(k):
        result = bbox_from_attention_adaptive(
            working,
            image_size,
            bbox_size=bbox_size,
        )
        results.append(result)
        left, top, right, bottom = result.bbox
        for row in range(working.shape[0]):
            center_y = (row + 0.5) * cell_height
            for column in range(working.shape[1]):
                center_x = (column + 0.5) * cell_width
                if left <= center_x <= right and top <= center_y <= bottom:
                    working[row, column] = suppression_value
    return results


def save_relative_attention_crop(
    image_path: Path,
    output_dir: Path,
    result: RelativeAttentionResult,
    *,
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

    np.save(output_dir / "relative_attention.npy", result.attention_map)
    np.save(output_dir / "task_attention.npy", result.task_attention)
    np.save(output_dir / "general_attention.npy", result.general_attention)
    metadata = {
        "source_image": _portable_source_path(image_path),
        "attention_layer": result.layer,
        "grid_shape": list(result.grid_shape),
        "image_token_span": [
            result.image_token_start,
            result.image_token_end,
        ],
        "general_prompt": GENERAL_PROMPT,
        "bbox_size": bbox_size,
        "crop": asdict(crop_result),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return crop_path, crop_result
