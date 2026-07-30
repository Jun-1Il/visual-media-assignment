#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicrop_mlx.dataset import build_samples
from vicrop_mlx.gradient_maps import (
    gradient_weighted_attention_qwen3_vl,
    input_gradient_qwen3_vl,
    save_gradient_map_crop,
)
from vicrop_mlx.rel_att import configure_author_qwen_resolution


def save_grayscale_map(values: np.ndarray, output_path: Path) -> None:
    values = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(values, [2, 98])
    if high <= low:
        normalized = np.zeros_like(values)
    else:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    image = Image.fromarray(np.uint8(normalized * 255), mode="L")
    image = image.resize(
        (max(256, image.width * 32), max(256, image.height * 32)),
        Image.Resampling.NEAREST,
    )
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect MLX grad-att or pure-grad on one Qwen3-VL sample."
    )
    parser.add_argument("sample_id")
    parser.add_argument(
        "--method",
        choices=("grad_att", "pure_grad", "pure_grad_repo"),
        required=True,
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-VL-2B-Instruct-4bit",
    )
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--validate-gradient", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "gradient_debug",
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
    max_pixels = configure_author_qwen_resolution(processor, model)
    if args.method == "grad_att":
        result = gradient_weighted_attention_qwen3_vl(
            model,
            processor,
            sample.image_path,
            prompt,
            layer=args.layer,
            validate_gradient=args.validate_gradient,
        )
    else:
        repository_variant = args.method == "pure_grad_repo"
        result = input_gradient_qwen3_vl(
            model,
            processor,
            sample.image_path,
            prompt,
            normalize_by_general=repository_variant,
            median_kernel=7 if repository_variant else 3,
            validate_gradient=args.validate_gradient,
        )

    sample_dir = (
        args.output_dir
        / args.method
        / sample.sample_id.replace("/", "__")
    )
    crop_path, crop = save_gradient_map_crop(
        sample.image_path,
        sample_dir,
        result,
        method=args.method,
    )
    save_grayscale_map(result.attention_map, sample_dir / "importance_map.png")
    print(f"sample={sample.sample_id}")
    print(f"method={args.method}")
    print(f"max_pixels={max_pixels}")
    print(f"grid={result.grid_shape}")
    print(f"target_token_id={result.target_token_id}")
    print(f"target_log_probability={result.target_log_probability:.8f}")
    print(f"bbox={crop.bbox}")
    print(f"ratio={crop.ratio}")
    print(f"difference={crop.attention_difference:.8f}")
    print(f"finite_difference={result.finite_difference}")
    print(f"crop={crop_path}")

    if args.generate:
        from mlx_vlm import apply_chat_template, generate

        for name, images in (
            ("original", [str(sample.image_path)]),
            ("original_plus_crop", [str(sample.image_path), str(crop_path)]),
        ):
            formatted = apply_chat_template(
                processor,
                model.config,
                prompt,
                num_images=len(images),
            )
            answer = generate(
                model,
                processor,
                formatted,
                image=images,
                max_tokens=args.max_tokens,
                temperature=0.0,
                verbose=False,
            )
            print(f"{name}_answer={answer.text!r}")


if __name__ == "__main__":
    main()
