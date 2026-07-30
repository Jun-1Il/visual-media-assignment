#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicrop_mlx.dataset import build_samples, write_manifest
from vicrop_mlx.evaluation import (
    load_rows,
    normalize_prediction,
    summarize,
    write_rows,
)
from vicrop_mlx.gradient_maps import (
    gradient_weighted_attention_qwen3_vl,
    input_gradient_qwen3_vl,
    save_gradient_map_crop,
)
from vicrop_mlx.high_resolution import (
    high_resolution_importance_map_qwen3_vl,
    save_high_resolution_crop,
)
from vicrop_mlx.rel_att import (
    GENERAL_PROMPT,
    configure_author_qwen_resolution,
    ensemble_log_relative_attention,
    relative_attention_layers_qwen3_vl,
    relative_attention_qwen3_vl,
    save_relative_attention_crop,
    topk_bboxes_from_attention,
)


def experiment_prompt(question: str) -> str:
    return f"{question} Answer the question using a single word or phrase."


def run_generation(
    model,
    processor,
    prompt: str,
    image_paths: list[Path],
    max_tokens: int,
):
    from mlx_vlm import apply_chat_template, generate

    formatted = apply_chat_template(
        processor,
        model.config,
        prompt,
        num_images=len(image_paths),
    )
    started = time.perf_counter()
    result = generate(
        model,
        processor,
        formatted,
        image=[str(path) for path in image_paths],
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    )
    return result, time.perf_counter() - started


def save_ensemble_crops(
    image_path: Path,
    output_dir: Path,
    attention_map: np.ndarray,
    crop_count: int,
    layers: tuple[int, ...],
    grid_shape: tuple[int, int],
) -> tuple[list[Path], list[tuple[int, int, int, int]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        crop_results = topk_bboxes_from_attention(
            attention_map,
            image.size,
            k=crop_count,
        )
        paths = []
        for index, crop_result in enumerate(crop_results, start=1):
            crop_path = output_dir / f"crop_{index}.png"
            image.crop(crop_result.bbox).save(crop_path)
            paths.append(crop_path)
    np.save(output_dir / "ensemble_log_relative_attention.npy", attention_map)
    metadata = {
        "source_image": image_path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "method": "median-MAD standardized log rel-att layer ensemble",
        "layers": list(layers),
        "grid_shape": list(grid_shape),
        "general_prompt": GENERAL_PROMPT,
        "crops": [
            {
                "bbox": list(crop_result.bbox),
                "ratio": crop_result.ratio,
                "window_blocks": list(crop_result.window_blocks),
                "attention_difference": crop_result.attention_difference,
            }
            for crop_result in crop_results
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths, [crop_result.bbox for crop_result in crop_results]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run all internal ViCrop methods on MLX Qwen3-VL: "
            "rel-att, grad-att, and pure-grad."
        )
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-VL-2B-Instruct-4bit",
    )
    parser.add_argument(
        "--variants",
        default=(
            "baseline,rel_att,grad_att,pure_grad,pure_grad_repo,"
            "rel_att_ensemble,rel_att_plus"
        ),
        help=(
            "Comma-separated: baseline,rel_att,grad_att,pure_grad,"
            "pure_grad_repo,rel_att_ensemble,rel_att_plus and optional "
            "*_high paper high-resolution variants"
        ),
    )
    parser.add_argument("--domains", default="")
    parser.add_argument("--conditions", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument(
        "--high-resolution-threshold",
        type=int,
        default=1024,
        help="Maximum width and height of each paper high-resolution block.",
    )
    parser.add_argument(
        "--layer-selection",
        type=Path,
        default=ROOT / "results" / "layer_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "internal_methods_predictions.csv",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    variants = tuple(
        item.strip() for item in args.variants.split(",") if item.strip()
    )
    valid_variants = {
        "baseline",
        "rel_att",
        "grad_att",
        "pure_grad",
        "pure_grad_repo",
        "rel_att_ensemble",
        "rel_att_plus",
        "rel_att_high",
        "grad_att_high",
        "pure_grad_high",
        "pure_grad_repo_high",
    }
    unknown = set(variants) - valid_variants
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")

    selection = json.loads(args.layer_selection.read_text(encoding="utf-8"))
    selected_layer = int(selection["selected_layer"])
    ensemble_layers = tuple(int(layer) for layer in selection["ensemble_layers"])
    calibration_ids = set(selection["calibration_sample_ids"])

    domains = {item.strip() for item in args.domains.split(",") if item.strip()}
    conditions = {
        item.strip() for item in args.conditions.split(",") if item.strip()
    }
    samples = [
        sample
        for sample in build_samples(ROOT)
        if (not domains or sample.domain in domains)
        and (not conditions or sample.condition in conditions)
    ]
    if args.limit is not None:
        samples = samples[: args.limit]
    write_manifest(build_samples(ROOT), ROOT, ROOT / "data" / "manifest.csv")

    rows = [] if args.no_resume else load_rows(args.output)
    completed = {(row["sample_id"], row["variant"]) for row in rows}

    from mlx_vlm import load

    print(f"Loading {args.model} ...", flush=True)
    model, processor = load(args.model)
    max_pixels = configure_author_qwen_resolution(processor, model)
    print(
        f"selected_layer={selected_layer}, ensemble_layers={ensemble_layers}, "
        f"max_pixels={max_pixels}",
        flush=True,
    )

    for sample_index, sample in enumerate(samples, start=1):
        prompt = experiment_prompt(sample.question)
        for variant in variants:
            key = (sample.sample_id, variant)
            if key in completed:
                continue

            attention_elapsed = 0.0
            bboxes: list[tuple[int, int, int, int]] = []
            grid_shape: tuple[int, int] | None = None
            attention_dir = (
                ROOT
                / "results"
                / "rel_att"
                / variant
                / sample.sample_id.replace("/", "__")
            )

            if variant == "baseline":
                generation_images = [sample.image_path]
            elif variant == "rel_att" or (
                variant == "rel_att_plus" and sample.task != "relation"
            ):
                attention_started = time.perf_counter()
                attention_result = relative_attention_qwen3_vl(
                    model,
                    processor,
                    sample.image_path,
                    prompt,
                    layer=selected_layer,
                )
                attention_elapsed = time.perf_counter() - attention_started
                crop_path, crop_result = save_relative_attention_crop(
                    sample.image_path,
                    attention_dir,
                    attention_result,
                )
                generation_images = [sample.image_path, crop_path]
                bboxes = [crop_result.bbox]
                grid_shape = attention_result.grid_shape
            elif variant == "grad_att":
                attention_started = time.perf_counter()
                attention_result = gradient_weighted_attention_qwen3_vl(
                    model,
                    processor,
                    sample.image_path,
                    prompt,
                    layer=selected_layer,
                )
                attention_elapsed = time.perf_counter() - attention_started
                crop_path, crop_result = save_gradient_map_crop(
                    sample.image_path,
                    attention_dir,
                    attention_result,
                    method=variant,
                )
                generation_images = [sample.image_path, crop_path]
                bboxes = [crop_result.bbox]
                grid_shape = attention_result.grid_shape
            elif variant in {"pure_grad", "pure_grad_repo"}:
                repository_variant = variant == "pure_grad_repo"
                attention_started = time.perf_counter()
                attention_result = input_gradient_qwen3_vl(
                    model,
                    processor,
                    sample.image_path,
                    prompt,
                    normalize_by_general=repository_variant,
                    median_kernel=7 if repository_variant else 3,
                )
                attention_elapsed = time.perf_counter() - attention_started
                crop_path, crop_result = save_gradient_map_crop(
                    sample.image_path,
                    attention_dir,
                    attention_result,
                    method=variant,
                )
                generation_images = [sample.image_path, crop_path]
                bboxes = [crop_result.bbox]
                grid_shape = attention_result.grid_shape
            elif variant in {
                "rel_att_high",
                "grad_att_high",
                "pure_grad_high",
                "pure_grad_repo_high",
            }:
                base_method = variant.removesuffix("_high")
                attention_started = time.perf_counter()
                attention_result = high_resolution_importance_map_qwen3_vl(
                    model,
                    processor,
                    sample.image_path,
                    prompt,
                    method=base_method,
                    layer=selected_layer,
                    threshold=args.high_resolution_threshold,
                )
                attention_elapsed = time.perf_counter() - attention_started
                crop_path, crop_result = save_high_resolution_crop(
                    sample.image_path,
                    attention_dir,
                    attention_result,
                )
                generation_images = [sample.image_path, crop_path]
                bboxes = [crop_result.bbox]
                grid_shape = attention_result.importance_map.shape
            else:
                attention_started = time.perf_counter()
                layer_results = relative_attention_layers_qwen3_vl(
                    model,
                    processor,
                    sample.image_path,
                    prompt,
                    ensemble_layers,
                )
                ensemble_map = ensemble_log_relative_attention(
                    layer_results,
                    ensemble_layers,
                )
                attention_elapsed = time.perf_counter() - attention_started
                crop_count = 2 if sample.task == "relation" else 1
                crop_paths, bboxes = save_ensemble_crops(
                    sample.image_path,
                    attention_dir,
                    ensemble_map,
                    crop_count,
                    ensemble_layers,
                    layer_results[ensemble_layers[0]].grid_shape,
                )
                generation_images = [sample.image_path, *crop_paths]
                grid_shape = layer_results[ensemble_layers[0]].grid_shape

            generation, generation_elapsed = run_generation(
                model,
                processor,
                prompt,
                generation_images,
                args.max_tokens,
            )
            prediction = normalize_prediction(generation.text, sample.task)
            correct = int(prediction == sample.answer)
            elapsed = attention_elapsed + generation_elapsed
            row = {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path.relative_to(ROOT).as_posix(),
                "domain": sample.domain,
                "condition": sample.condition,
                "task": sample.task,
                "question": sample.question,
                "answer": sample.answer,
                "variant": variant,
                "prediction": prediction,
                "raw_response": generation.text.strip().replace("\n", "\\n"),
                "correct": correct,
                "is_calibration": int(sample.sample_id in calibration_ids),
                "attention_layer": (
                    selected_layer
                    if variant == "rel_att"
                    or variant == "grad_att"
                    or variant == "rel_att_high"
                    or variant == "grad_att_high"
                    or (variant == "rel_att_plus" and sample.task != "relation")
                    else ""
                ),
                "ensemble_layers": (
                    " ".join(map(str, ensemble_layers))
                    if variant == "rel_att_ensemble"
                    or (variant == "rel_att_plus" and sample.task == "relation")
                    else ""
                ),
                "grid_shape": (
                    f"{grid_shape[0]}x{grid_shape[1]}" if grid_shape else ""
                ),
                "bbox_1": " ".join(map(str, bboxes[0])) if bboxes else "",
                "bbox_2": (
                    " ".join(map(str, bboxes[1])) if len(bboxes) > 1 else ""
                ),
                "n_images": len(generation_images),
                "attention_s": f"{attention_elapsed:.6f}",
                "generation_s": f"{generation_elapsed:.6f}",
                "elapsed_s": f"{elapsed:.6f}",
                "prompt_tokens": generation.prompt_tokens,
                "generation_tokens": generation.generation_tokens,
                "prompt_tps": f"{generation.prompt_tps:.3f}",
                "generation_tps": f"{generation.generation_tps:.3f}",
                "peak_memory_gb": f"{generation.peak_memory:.6f}",
                "max_pixels": max_pixels,
                "model": args.model,
            }
            rows.append(row)
            completed.add(key)
            write_rows(args.output, rows)
            mark = "OK" if correct else "MISS"
            split = "CAL" if sample.sample_id in calibration_ids else "TEST"
            print(
                f"[{sample_index:03d}/{len(samples):03d} {variant}] "
                f"{mark}/{split} {sample.sample_id}: {prediction!r} "
                f"(gt={sample.answer}, {elapsed:.2f}s)",
                flush=True,
            )

    write_rows(
        ROOT / "results" / "internal_methods_metrics_by_condition.csv",
        summarize(rows),
    )
    test_rows = [row for row in rows if int(row["is_calibration"]) == 0]
    payload = {
        "model": args.model,
        "max_pixels": max_pixels,
        "selected_layer": selected_layer,
        "ensemble_layers": list(ensemble_layers),
        "calibration_sample_ids": sorted(calibration_ids),
        "all_91": {
            row["variant"]: {
                "correct": sum(
                    int(candidate["correct"])
                    for candidate in rows
                    if candidate["variant"] == row["variant"]
                ),
                "count": sum(
                    candidate["variant"] == row["variant"] for candidate in rows
                ),
            }
            for row in rows
        },
        "held_out_test": {
            row["variant"]: {
                "correct": sum(
                    int(candidate["correct"])
                    for candidate in test_rows
                    if candidate["variant"] == row["variant"]
                ),
                "count": sum(
                    candidate["variant"] == row["variant"]
                    for candidate in test_rows
                ),
            }
            for row in test_rows
        },
    }
    (ROOT / "results" / "internal_methods_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
