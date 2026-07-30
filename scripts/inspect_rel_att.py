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
from vicrop_mlx.rel_att import (
    AUTHOR_QWEN_ATTENTION_LAYER,
    configure_author_qwen_resolution,
    relative_attention_qwen3_vl,
    save_relative_attention_crop,
)


def save_grayscale_map(attention_map: np.ndarray, output_path: Path) -> None:
    values = np.asarray(attention_map, dtype=np.float64)
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
        description="Extract one strict Qwen3-VL relative-attention map with MLX."
    )
    parser.add_argument("sample_id")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-VL-2B-Instruct-4bit",
    )
    parser.add_argument("--layer", type=int, default=AUTHOR_QWEN_ATTENTION_LAYER)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "rel_att_debug",
    )
    args = parser.parse_args()

    samples = {sample.sample_id: sample for sample in build_samples(ROOT)}
    sample = samples[args.sample_id]

    from mlx_vlm import load

    model, processor = load(args.model)
    max_pixels = configure_author_qwen_resolution(processor, model)
    question = (
        f"{sample.question} "
        "Answer the question using a single word or phrase."
    )
    result = relative_attention_qwen3_vl(
        model,
        processor,
        sample.image_path,
        question,
        layer=args.layer,
    )
    sample_dir = args.output_dir / sample.sample_id.replace("/", "__")
    crop_path, crop = save_relative_attention_crop(
        sample.image_path,
        sample_dir,
        result,
    )
    save_grayscale_map(result.attention_map, sample_dir / "relative_attention.png")
    print(f"sample={sample.sample_id}")
    print(f"max_pixels={max_pixels}")
    print(f"layer={result.layer}")
    print(f"grid={result.grid_shape}")
    print(
        "image_token_span="
        f"[{result.image_token_start}, {result.image_token_end})"
    )
    print(f"bbox={crop.bbox}")
    print(f"ratio={crop.ratio}")
    print(f"difference={crop.attention_difference:.8f}")
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
                question,
                num_images=len(images),
            )
            answer = generate(
                model,
                processor,
                formatted,
                image=images,
                max_tokens=20,
                temperature=0.0,
                verbose=False,
            )
            print(f"{name}_answer={answer.text!r}")


if __name__ == "__main__":
    main()
