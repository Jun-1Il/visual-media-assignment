#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicrop_mlx.dataset import build_samples
from vicrop_mlx.gradient_maps import (
    _DifferentiableAttention,
    _pixel_gradient,
    _prepared_forward_state,
    _target_log_probability,
    gradient_weighted_attention_qwen3_vl,
)
from vicrop_mlx.rel_att import configure_author_qwen_resolution


def relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.linalg.norm(first.astype(np.float64) - second.astype(np.float64))
    reference = max(np.linalg.norm(first.astype(np.float64)), 1e-12)
    return float(difference / reference)


def grad_attention_validation(
    model,
    processor,
    image_path: Path,
    prompt: str,
    *,
    layer: int,
) -> dict[str, object]:
    import mlx.core as mx

    state = _prepared_forward_state(model, processor, image_path, prompt)
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
    reference_logits = model.language_model(
        state["input_ids"],
        inputs_embeds=embedding_output.inputs_embeds,
        **language_kwargs,
    ).logits
    layers = model.language_model.model.layers
    original = layers[layer].self_attn
    wrapper = _DifferentiableAttention(original)
    layers[layer].self_attn = wrapper
    length = int(state["input_ids"].shape[-1])
    zero = mx.zeros((1, original.n_heads, 1, length), dtype=mx.float32)
    wrapper.final_row_delta = zero
    try:
        explicit_logits = model.language_model(
            state["input_ids"],
            inputs_embeds=embedding_output.inputs_embeds,
            **language_kwargs,
        ).logits
        mx.eval(reference_logits, explicit_logits)
    finally:
        layers[layer].self_attn = original
        mx.clear_cache()

    reference = np.asarray(
        reference_logits[0, -1].astype(mx.float32).tolist(),
        dtype=np.float32,
    )
    explicit = np.asarray(
        explicit_logits[0, -1].astype(mx.float32).tolist(),
        dtype=np.float32,
    )
    result = gradient_weighted_attention_qwen3_vl(
        model,
        processor,
        image_path,
        prompt,
        layer=layer,
    )
    reconstructed = (
        result.attention * np.maximum(result.attention_gradient, 0.0)
    ).mean(axis=0).reshape(result.grid_shape)
    return {
        "grid_shape": list(result.grid_shape),
        "reference_argmax": int(np.argmax(reference)),
        "explicit_argmax": int(np.argmax(explicit)),
        "argmax_identical": bool(np.argmax(reference) == np.argmax(explicit)),
        "final_logits_relative_l2_error": relative_l2(reference, explicit),
        "map_formula_max_abs_error": float(
            np.max(np.abs(reconstructed - result.attention_map))
        ),
        "attention_gradient_finite": bool(
            np.isfinite(result.attention_gradient).all()
        ),
        "attention_gradient_nonzero": bool(
            np.any(result.attention_gradient != 0)
        ),
    }


def pure_gradient_chain_validation(
    model,
    processor,
    image_path: Path,
    prompt: str,
) -> dict[str, object]:
    """Cross-check the Conv3d-equivalent direct pixel-gradient implementation."""

    import mlx.core as mx

    state = _prepared_forward_state(model, processor, image_path, prompt)
    implementation_gradient, target_token_id, _, _ = _pixel_gradient(
        model,
        state,
        validate_gradient=False,
    )
    vision = model.vision_tower
    patch_embed = vision.patch_embed
    grid_thw = state["image_grid_thw"]
    pixel_values = state["pixel_values"].astype(mx.float32)
    weight = patch_embed.proj.weight.astype(mx.float32)
    bias = patch_embed.proj.bias.astype(mx.float32)
    vision_dtype = patch_embed.proj.weight.dtype

    def manual_patch_embeddings(pixels):
        convolution_order = (
            pixels.reshape(
                -1,
                patch_embed.in_channels,
                patch_embed.temporal_patch_size,
                patch_embed.patch_size,
                patch_embed.patch_size,
            )
            .moveaxis(1, 4)
            .reshape(pixels.shape[0], -1)
        )
        return (
            convolution_order @ weight.reshape(weight.shape[0], -1).T + bias
        ).astype(vision_dtype)

    def vision_outputs_from_patch_embeddings(patch_embeddings):
        hidden_states = patch_embeddings + vision.fast_pos_embed_interpolate(
            grid_thw
        )
        rotary_pos_emb = vision.rot_pos_emb(grid_thw)
        sequence_length = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(sequence_length, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(sequence_length, -1)
        cumulative_lengths = []
        for index in range(grid_thw.shape[0]):
            sequence = grid_thw[index, 1] * grid_thw[index, 2]
            cumulative_lengths.append(mx.repeat(sequence, grid_thw[index, 0]))
        cumulative_lengths = mx.concatenate(cumulative_lengths)
        cumulative_lengths = mx.pad(
            mx.cumsum(cumulative_lengths.astype(mx.int32), axis=0),
            (1, 0),
            mode="constant",
            constant_values=0,
        )
        deepstack = []
        for layer_number, block in enumerate(vision.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cumulative_lengths,
                rotary_pos_emb=rotary_pos_emb,
            )
            if layer_number in vision.deepstack_visual_indexes:
                merger_index = vision.deepstack_visual_indexes.index(layer_number)
                deepstack.append(
                    vision.deepstack_merger_list[merger_index](hidden_states)
                )
        return [vision.merger(hidden_states), *deepstack]

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
    native_patch_anchor = mx.stop_gradient(
        patch_embed(state["pixel_values"].astype(vision_dtype))
    )
    manual_patch_anchor = mx.stop_gradient(
        manual_patch_embeddings(pixel_values)
    )

    def direct_objective(pixels):
        # Evaluate at the exact native Conv3d forward point while using the
        # analytically identical linear patch-embed Jacobian.  This removes
        # only Conv3d-vs-matmul accumulation-order rounding from the check.
        patch_embeddings = (
            native_patch_anchor
            + (manual_patch_embeddings(pixels) - manual_patch_anchor)
        )
        main, *deepstack = vision_outputs_from_patch_embeddings(
            patch_embeddings
        )
        inputs_embeds = mx.concatenate(
            [
                text_embeddings[:, :image_start, :],
                main[None, :, :],
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
            deepstack_visual_embeds=deepstack,
        ).logits
        return _target_log_probability(logits, target_token_id)

    direct = mx.grad(direct_objective)(pixel_values)
    native_patch = native_patch_anchor
    manual_patch = manual_patch_embeddings(pixel_values)
    mx.eval(direct, native_patch, manual_patch)
    direct_np = np.asarray(
        direct.astype(mx.float32).tolist(),
        dtype=np.float32,
    )
    native_np = np.asarray(
        native_patch.astype(mx.float32).tolist(),
        dtype=np.float32,
    )
    manual_np = np.asarray(
        manual_patch.astype(mx.float32).tolist(),
        dtype=np.float32,
    )
    return {
        "grid_thw": state["grid"].tolist(),
        "pixel_values_shape": list(implementation_gradient.shape),
        "target_token_id": target_token_id,
        "patch_embed_forward_relative_l2_error": relative_l2(
            native_np,
            manual_np,
        ),
        "implementation_vs_validation_path_gradient_relative_l2_error": relative_l2(
            direct_np,
            implementation_gradient,
        ),
        "implementation_vs_validation_path_gradient_max_abs_error": float(
            np.max(np.abs(direct_np - implementation_gradient))
        ),
        "validation_path_gradient_finite": bool(np.isfinite(direct_np).all()),
        "validation_path_gradient_nonzero": bool(np.any(direct_np != 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate MLX grad-att and whole-graph pure-grad internals."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-VL-2B-Instruct-4bit",
    )
    parser.add_argument(
        "--sample-id",
        default="synthetic/000_tiny_clear",
    )
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument(
        "--validation-pixels",
        type=int,
        default=4096,
        help="Small square pixel budget used for the direct gradient cross-check.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "gradient_internal_validation.json",
    )
    args = parser.parse_args()

    samples = {sample.sample_id: sample for sample in build_samples(ROOT)}
    sample = samples[args.sample_id]
    prompt = (
        f"{sample.question} "
        "Answer the question using a single word or phrase."
    )
    from mlx_vlm import load

    model, processor = load(args.model)
    author_max_pixels = configure_author_qwen_resolution(processor, model)
    grad_attention = grad_attention_validation(
        model,
        processor,
        sample.image_path,
        prompt,
        layer=args.layer,
    )

    image_processor = processor.image_processor
    old_min = image_processor.min_pixels
    old_max = image_processor.max_pixels
    image_processor.min_pixels = args.validation_pixels
    image_processor.max_pixels = args.validation_pixels
    try:
        pure_gradient = pure_gradient_chain_validation(
            model,
            processor,
            sample.image_path,
            prompt,
        )
    finally:
        image_processor.min_pixels = old_min
        image_processor.max_pixels = old_max

    payload = {
        "model": args.model,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
        "sample_id": sample.sample_id,
        "layer": args.layer,
        "author_max_pixels": author_max_pixels,
        "gradient_crosscheck_pixels": args.validation_pixels,
        "grad_attention": grad_attention,
        "pure_gradient": pure_gradient,
        "criteria": {
            "grad_attention_argmax_identical": True,
            "grad_attention_map_formula_max_abs_error_at_most": 1e-12,
            "pure_gradient_patch_embed_relative_l2_error_at_most": 0.01,
            "pure_gradient_implementation_vs_validation_path_relative_l2_error_at_most": 0.01,
        },
    }
    payload["passed"] = bool(
        grad_attention["argmax_identical"]
        and grad_attention["map_formula_max_abs_error"] <= 1e-12
        and pure_gradient["patch_embed_forward_relative_l2_error"] <= 0.01
        and pure_gradient[
            "implementation_vs_validation_path_gradient_relative_l2_error"
        ]
        <= 0.01
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit("Gradient internal validation failed")


if __name__ == "__main__":
    main()
