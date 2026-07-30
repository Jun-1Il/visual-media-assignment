#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicrop_mlx.crops import find_colored_bbox
from vicrop_mlx.dataset import build_samples
from vicrop_mlx.evaluation import normalize_prediction
from vicrop_mlx.rel_att import (
    bbox_from_attention_adaptive,
    configure_author_qwen_resolution,
    relative_attention_layers_qwen3_vl,
)


CALIBRATION_SAMPLE_IDS = (
    "synthetic/000_small_clear",
    "synthetic/000_tiny_clear",
    "synthetic/001_small_clear",
    "synthetic/001_tiny_clear",
    "synthetic/008_large_clear",
    "synthetic/009_large_clear",
    "synthetic/016_small_low_contrast",
    "synthetic/017_small_low_contrast",
)


def run_generation(model, processor, prompt: str, images: list[Path], max_tokens: int):
    from mlx_vlm import apply_chat_template, generate

    formatted = apply_chat_template(
        processor,
        model.config,
        prompt,
        num_images=len(images),
    )
    started = time.perf_counter()
    result = generate(
        model,
        processor,
        formatted,
        image=[str(path) for path in images],
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    )
    return result, time.perf_counter() - started


def intersection_over_target(
    crop: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
) -> float:
    left = max(crop[0], target[0])
    top = max(crop[1], target[1])
    right = min(crop[2], target[2])
    bottom = min(crop[3], target[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    target_area = max(1, target[2] - target[0]) * max(1, target[3] - target[1])
    return intersection / target_area


def center_distance(
    crop: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> float:
    crop_x = (crop[0] + crop[2]) / 2
    crop_y = (crop[1] + crop[3]) / 2
    target_x = (target[0] + target[2]) / 2
    target_y = (target[1] + target[3]) / 2
    diagonal = (image_size[0] ** 2 + image_size[1] ** 2) ** 0.5
    return ((crop_x - target_x) ** 2 + (crop_y - target_y) ** 2) ** 0.5 / diagonal


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select Qwen3-VL rel-att layers on a fixed held-out calibration set."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-VL-2B-Instruct-4bit",
    )
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "layer_calibration.csv",
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=ROOT / "results" / "layer_selection.json",
    )
    args = parser.parse_args()

    sample_lookup = {sample.sample_id: sample for sample in build_samples(ROOT)}
    samples = [sample_lookup[sample_id] for sample_id in CALIBRATION_SAMPLE_IDS]

    from mlx_vlm import load

    model, processor = load(args.model)
    max_pixels = configure_author_qwen_resolution(processor, model)
    layers = tuple(range(len(model.language_model.model.layers)))
    rows: list[dict[str, object]] = []

    for sample_index, sample in enumerate(samples, start=1):
        prompt = (
            f"{sample.question} "
            "Answer the question using a single word or phrase."
        )
        maps = relative_attention_layers_qwen3_vl(
            model,
            processor,
            sample.image_path,
            prompt,
            layers,
        )
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
            image_size = image.size
            target = find_colored_bbox(image, "red")
            for layer in layers:
                crop_result = bbox_from_attention_adaptive(
                    maps[layer].attention_map,
                    image_size,
                )
                crop_dir = (
                    ROOT
                    / "results"
                    / "calibration_crops"
                    / sample.sample_id.replace("/", "__")
                )
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_path = crop_dir / f"layer_{layer:02d}.png"
                image.crop(crop_result.bbox).save(crop_path)
                generation, elapsed = run_generation(
                    model,
                    processor,
                    prompt,
                    [sample.image_path, crop_path],
                    args.max_tokens,
                )
                prediction = normalize_prediction(generation.text, sample.task)
                coverage = intersection_over_target(crop_result.bbox, target)
                target_center = (
                    (target[0] + target[2]) / 2,
                    (target[1] + target[3]) / 2,
                )
                center_hit = int(
                    crop_result.bbox[0] <= target_center[0] <= crop_result.bbox[2]
                    and crop_result.bbox[1]
                    <= target_center[1]
                    <= crop_result.bbox[3]
                )
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "layer": layer,
                        "grid_h": maps[layer].grid_shape[0],
                        "grid_w": maps[layer].grid_shape[1],
                        "bbox": " ".join(map(str, crop_result.bbox)),
                        "ratio": crop_result.ratio,
                        "attention_difference": (
                            f"{crop_result.attention_difference:.9f}"
                        ),
                        "target_coverage": f"{coverage:.9f}",
                        "target_center_hit": center_hit,
                        "center_distance": (
                            f"{center_distance(crop_result.bbox, target, image_size):.9f}"
                        ),
                        "prediction": prediction,
                        "answer": sample.answer,
                        "correct": int(prediction == sample.answer),
                        "raw_response": generation.text.strip().replace("\n", "\\n"),
                        "elapsed_s": f"{elapsed:.6f}",
                    }
                )
        write_csv(args.output, rows)
        print(
            f"[{sample_index}/{len(samples)}] {sample.sample_id}: "
            f"{len(layers)} layers",
            flush=True,
        )

    by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer"])].append(row)
    summaries = []
    for layer, layer_rows in sorted(by_layer.items()):
        summaries.append(
            {
                "layer": layer,
                "correct": sum(int(row["correct"]) for row in layer_rows),
                "count": len(layer_rows),
                "center_hits": sum(
                    int(row["target_center_hit"]) for row in layer_rows
                ),
                "mean_target_coverage": sum(
                    float(row["target_coverage"]) for row in layer_rows
                )
                / len(layer_rows),
                "mean_center_distance": sum(
                    float(row["center_distance"]) for row in layer_rows
                )
                / len(layer_rows),
            }
        )
    ranking = sorted(
        summaries,
        key=lambda row: (
            -row["correct"],
            -row["center_hits"],
            -row["mean_target_coverage"],
            row["mean_center_distance"],
            row["layer"],
        ),
    )
    payload = {
        "model": args.model,
        "max_pixels": max_pixels,
        "visual_token_budget": 256,
        "selection_protocol": (
            "Rank by exact-match accuracy, target-center hits, target coverage, "
            "then center distance on the fixed calibration set."
        ),
        "calibration_sample_ids": list(CALIBRATION_SAMPLE_IDS),
        "selected_layer": ranking[0]["layer"],
        "ensemble_layers": [row["layer"] for row in ranking[:3]],
        "ranking": ranking,
    }
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"selected_layer={payload['selected_layer']} "
        f"ensemble_layers={payload['ensemble_layers']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
