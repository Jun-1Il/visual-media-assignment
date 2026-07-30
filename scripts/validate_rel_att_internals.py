#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicrop_mlx.dataset import build_samples
from vicrop_mlx.rel_att import (
    GENERAL_PROMPT,
    configure_author_qwen_resolution,
    validate_attention_reconstruction_qwen3_vl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate explicit Q/K attention reconstruction against the "
            "loaded MLX fused-SDPA layer output."
        )
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
        "--max-relative-l2-error",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "attention_reconstruction_validation.json",
    )
    args = parser.parse_args()

    samples = {sample.sample_id: sample for sample in build_samples(ROOT)}
    sample = samples[args.sample_id]
    task_prompt = (
        f"{sample.question} "
        "Answer the question using a single word or phrase."
    )

    from mlx_vlm import load

    model, processor = load(args.model)
    max_pixels = configure_author_qwen_resolution(processor, model)
    task = validate_attention_reconstruction_qwen3_vl(
        model,
        processor,
        sample.image_path,
        task_prompt,
        layer=args.layer,
    )
    general = validate_attention_reconstruction_qwen3_vl(
        model,
        processor,
        sample.image_path,
        GENERAL_PROMPT,
        layer=args.layer,
    )

    payload = {
        "model": args.model,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
        "sample_id": sample.sample_id,
        "layer": args.layer,
        "max_pixels": max_pixels,
        "criterion": {
            "relative_l2_error_at_most": args.max_relative_l2_error,
            "attention_row_sum_error_at_most": 1e-6,
        },
        "task_prompt": task_prompt,
        "general_prompt": GENERAL_PROMPT,
        "task": task,
        "general": general,
    }
    payload["passed"] = all(
        result["relative_l2_error"] <= args.max_relative_l2_error
        and result["attention_row_sum_error"] <= 1e-6
        for result in (task, general)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit("Attention reconstruction validation failed")


if __name__ == "__main__":
    main()
