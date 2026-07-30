#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_rel_att_results import (  # noqa: E402
    bbox_on_image,
    center_inside,
    find_colored_bbox,
    font,
    heatmap_gray,
    load_gray,
    panel_figure,
    parse_bbox,
    plot_layer_calibration,
)


VARIANTS = (
    "baseline",
    "rel_att",
    "grad_att",
    "pure_grad",
    "pure_grad_repo",
    "rel_att_high",
    "grad_att_high",
    "pure_grad_high",
    "pure_grad_repo_high",
    "rel_att_ensemble",
    "rel_att_plus",
)
CORE_VARIANTS = ("baseline", "rel_att", "grad_att", "pure_grad")
LABELS = {
    "baseline": "Original",
    "rel_att": "rel-att",
    "grad_att": "grad-att",
    "pure_grad": "pure-grad (paper)",
    "pure_grad_repo": "pure-grad (repository)",
    "rel_att_high": "rel-att high-res",
    "grad_att_high": "grad-att high-res",
    "pure_grad_high": "pure-grad high-res",
    "pure_grad_repo_high": "repo pure-grad high-res",
    "rel_att_ensemble": "rel-att ensemble",
    "rel_att_plus": "rel-att+",
}
CONDITIONS = (
    "large_clear",
    "small_clear",
    "tiny_clear",
    "small_low_contrast",
    "multi_region_relation",
    "naturalistic_small_text",
)
CONDITION_LABELS = {
    "large_clear": "large",
    "small_clear": "small",
    "tiny_clear": "tiny",
    "small_low_contrast": "low contrast",
    "multi_region_relation": "two-region",
    "naturalistic_small_text": "natural",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def map_path(sample_id: str, variant: str) -> Path:
    directory = (
        ROOT
        / "results"
        / "rel_att"
        / variant
        / sample_id.replace("/", "__")
    )
    if variant.startswith("rel_att") and variant not in {
        "rel_att_high",
        "rel_att_ensemble",
        "rel_att_plus",
    }:
        return directory / "relative_attention.npy"
    if variant == "rel_att_high":
        return directory / "importance_map.npy"
    return directory / "importance_map.npy"


def write_condition_csv(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("variant", "condition", "correct", "count", "accuracy"),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_core_accuracy(
    test_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    width, height = 2300, 940
    left, top, right, bottom = 150, 120, 60, 185
    plot_width = width - left - right
    plot_height = height - top - bottom
    canvas = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (left, 35),
        "Held-out exact-match accuracy: all internal ViCrop maps",
        font=font(34, True),
        fill=0,
    )
    for tick in range(0, 101, 20):
        y = top + plot_height - round(plot_height * tick / 100)
        draw.line((left, y, left + plot_width, y), fill=220, width=2)
        draw.text((68, y - 15), str(tick), font=font(25), fill=0)
    draw.line((left, top, left, top + plot_height), fill=0, width=3)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill=0,
        width=3,
    )

    group_width = plot_width / len(CONDITIONS)
    bar_width = int(group_width * 0.16)
    fills = (235, 165, 85, 15)
    for condition_index, condition in enumerate(CONDITIONS):
        center = left + group_width * (condition_index + 0.5)
        for variant_index, variant in enumerate(CORE_VARIANTS):
            subset = [
                row
                for row in test_rows
                if row["variant"] == variant and row["condition"] == condition
            ]
            value = 100 * sum(int(row["correct"]) for row in subset) / len(subset)
            x0 = round(
                center
                + (variant_index - 1.5) * (bar_width + 7)
                - bar_width / 2
            )
            y0 = top + plot_height - round(plot_height * value / 100)
            draw.rectangle(
                (x0, y0, x0 + bar_width, top + plot_height),
                fill=fills[variant_index],
                outline=0,
                width=2,
            )
            if value:
                draw.text(
                    (x0 + 2, y0 - 30),
                    f"{value:.0f}",
                    font=font(18),
                    fill=0,
                )
        label = CONDITION_LABELS[condition]
        box = draw.textbbox((0, 0), label, font=font(23))
        draw.text(
            (
                center - (box[2] - box[0]) / 2,
                top + plot_height + 28,
            ),
            label,
            font=font(23),
            fill=0,
        )

    legend_left = left + 120
    for index, variant in enumerate(CORE_VARIANTS):
        x = legend_left + index * 475
        draw.rectangle(
            (x, 76, x + 42, 106),
            fill=fills[index],
            outline=0,
            width=2,
        )
        draw.text((x + 55, 72), LABELS[variant], font=font(22), fill=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def make_map_comparison(
    lookup: dict[tuple[str, str], dict[str, str]],
    output_dir: Path,
) -> None:
    sample_id = "synthetic/002_small_clear"
    original = ROOT / lookup[(sample_id, "baseline")]["image_path"]
    with Image.open(original) as source:
        target = find_colored_bbox(source.convert("RGB"), "red")
    panels = [
        (
            "Target (dashed)",
            bbox_on_image(original, [(target, "dashed")], (520, 560)),
        )
    ]
    for variant in ("rel_att", "grad_att", "pure_grad"):
        row = lookup[(sample_id, variant)]
        panels.append(
            (
                f"{LABELS[variant]}\noutput={row['prediction']}",
                heatmap_gray(
                    np.load(map_path(sample_id, variant)),
                    (520, 560),
                ),
            )
        )
    panel_figure(
        panels,
        output_dir / "internal_method_map_comparison.png",
        width=2400,
        panel_height=560,
    )

    sample_id = "synthetic/004_tiny_clear"
    original = ROOT / lookup[(sample_id, "baseline")]["image_path"]
    row = lookup[(sample_id, "rel_att")]
    directory = (
        ROOT
        / "results"
        / "rel_att"
        / "rel_att"
        / sample_id.replace("/", "__")
    )
    with Image.open(original) as source:
        target = find_colored_bbox(source.convert("RGB"), "red")
        detail = source.convert("L").crop(target)
    panel_figure(
        [
            (
                "Target hit by selected crop",
                bbox_on_image(
                    original,
                    [
                        (target, "dashed"),
                        (parse_bbox(row["bbox_1"]), "solid"),
                    ],
                    (660, 600),
                ),
            ),
            ("rel-att crop", load_gray(directory / "crop.png")),
            (
                f"Glyph detail\nGT={row['answer']} output={row['prediction']}",
                detail,
            ),
        ],
        output_dir / "failure_perception_after_localization.png",
    )

    sample_id = "synthetic/002_small_clear"
    original = ROOT / lookup[(sample_id, "baseline")]["image_path"]
    paper = lookup[(sample_id, "pure_grad")]
    repository = lookup[(sample_id, "pure_grad_repo")]
    panel_figure(
        [
            (
                "Paper / repository crops",
                bbox_on_image(
                    original,
                    [
                        (parse_bbox(paper["bbox_1"]), "solid"),
                        (parse_bbox(repository["bbox_1"]), "dashed"),
                    ],
                    (660, 600),
                ),
            ),
            (
                f"Paper pure-grad\noutput={paper['prediction']}",
                heatmap_gray(
                    np.load(map_path(sample_id, "pure_grad")),
                    (660, 600),
                ),
            ),
            (
                f"Repository variant\noutput={repository['prediction']}",
                heatmap_gray(
                    np.load(map_path(sample_id, "pure_grad_repo")),
                    (660, 600),
                ),
            ),
        ],
        output_dir / "pure_grad_paper_repository_difference.png",
    )

    sample_id = "synthetic/026_multi_region_relation"
    original = ROOT / lookup[(sample_id, "baseline")]["image_path"]
    strict = lookup[(sample_id, "rel_att")]
    plus = lookup[(sample_id, "rel_att_plus")]
    plus_dir = (
        ROOT
        / "results"
        / "rel_att"
        / "rel_att_plus"
        / sample_id.replace("/", "__")
    )
    crop_1 = load_gray(plus_dir / "crop_1.png")
    crop_2 = load_gray(plus_dir / "crop_2.png")
    pair = Image.new("L", (crop_1.width + crop_2.width, max(crop_1.height, crop_2.height)), 255)
    pair.paste(crop_1, (0, 0))
    pair.paste(crop_2, (crop_1.width, 0))
    panel_figure(
        [
            (
                "Two selected regions",
                bbox_on_image(
                    original,
                    [
                        (parse_bbox(plus["bbox_1"]), "solid"),
                        (parse_bbox(plus["bbox_2"]), "dashed"),
                    ],
                    (660, 600),
                ),
            ),
            (
                f"One-crop rel-att\noutput={strict['prediction']}",
                load_gray(
                    ROOT
                    / "results"
                    / "rel_att"
                    / "rel_att"
                    / sample_id.replace("/", "__")
                    / "crop.png"
                ),
            ),
            (
                f"Two-crop rel-att+\noutput={plus['prediction']}",
                pair,
            ),
        ],
        output_dir / "failure_relation_and_fix.png",
    )


def main() -> None:
    rows = read_rows(ROOT / "results" / "internal_methods_predictions.csv")
    duplicates = len(rows) - len(
        {(row["sample_id"], row["variant"]) for row in rows}
    )
    if duplicates:
        raise ValueError(f"Prediction table contains {duplicates} duplicate rows")
    test_rows = [row for row in rows if row["is_calibration"] == "0"]
    available = tuple(
        variant for variant in VARIANTS if any(row["variant"] == variant for row in rows)
    )
    lookup = {(row["sample_id"], row["variant"]): row for row in rows}
    sample_ids = sorted(
        {
            row["sample_id"]
            for row in test_rows
            if row["variant"] == "baseline"
        }
    )
    counts = {
        variant: sum(row["variant"] == variant for row in rows)
        for variant in available
    }
    if any(count != 91 for count in counts.values()):
        raise ValueError(f"Every available variant must contain 91 rows: {counts}")

    overall: dict[str, dict[str, float | int]] = {}
    by_condition: list[dict[str, object]] = []
    paired: dict[str, dict[str, int]] = {}
    for variant in available:
        subset = [row for row in test_rows if row["variant"] == variant]
        correct = sum(int(row["correct"]) for row in subset)
        overall[variant] = {
            "correct": correct,
            "count": len(subset),
            "accuracy": correct / len(subset),
            "mean_elapsed_s": statistics.mean(
                float(row["elapsed_s"]) for row in subset
            ),
            "mean_map_s": statistics.mean(
                float(row["attention_s"]) for row in subset
            ),
        }
        for condition in CONDITIONS:
            condition_rows = [
                row for row in subset if row["condition"] == condition
            ]
            condition_correct = sum(
                int(row["correct"]) for row in condition_rows
            )
            by_condition.append(
                {
                    "variant": variant,
                    "condition": condition,
                    "correct": condition_correct,
                    "count": len(condition_rows),
                    "accuracy": condition_correct / len(condition_rows),
                }
            )
        if variant != "baseline":
            paired[variant] = {
                "baseline_error_corrected": sum(
                    lookup[(sample_id, "baseline")]["correct"] == "0"
                    and lookup[(sample_id, variant)]["correct"] == "1"
                    for sample_id in sample_ids
                ),
                "baseline_correct_harmed": sum(
                    lookup[(sample_id, "baseline")]["correct"] == "1"
                    and lookup[(sample_id, variant)]["correct"] == "0"
                    for sample_id in sample_ids
                ),
            }

    localization = defaultdict(lambda: defaultdict(int))
    localization_variants = [
        variant for variant in available if variant != "baseline"
    ]
    for sample_id in sample_ids:
        baseline = lookup[(sample_id, "baseline")]
        if baseline["domain"] != "synthetic":
            continue
        with Image.open(ROOT / baseline["image_path"]) as source:
            image = source.convert("RGB")
            targets = [find_colored_bbox(image, "red")]
            if baseline["task"] == "relation":
                targets.append(find_colored_bbox(image, "blue"))
        for variant in localization_variants:
            row = lookup[(sample_id, variant)]
            crops = [
                bbox
                for bbox in (
                    parse_bbox(row["bbox_1"]),
                    parse_bbox(row["bbox_2"]),
                )
                if bbox is not None
            ]
            hits = sum(center_inside(target, crops) for target in targets)
            localization[variant]["target_centers"] += len(targets)
            localization[variant]["target_center_hits"] += hits
            localization[variant]["samples"] += 1
            localization[variant]["all_targets_hit"] += int(
                hits == len(targets)
            )
            if baseline["task"] == "code" and row["correct"] == "0":
                key = (
                    "wrong_after_localization"
                    if hits == 1
                    else "wrong_from_localization"
                )
                localization[variant][key] += 1

    core_oracle = sum(
        any(
            lookup[(sample_id, variant)]["correct"] == "1"
            for variant in ("rel_att", "grad_att", "pure_grad")
        )
        for sample_id in sample_ids
    )
    paper_repo_disagreements = sum(
        lookup[(sample_id, "pure_grad")]["correct"]
        != lookup[(sample_id, "pure_grad_repo")]["correct"]
        for sample_id in sample_ids
    )
    high_resolution_delta = {}
    for base in ("rel_att", "grad_att", "pure_grad", "pure_grad_repo"):
        high = f"{base}_high"
        if high in available:
            high_resolution_delta[base] = {
                "standard_correct": overall[base]["correct"],
                "high_resolution_correct": overall[high]["correct"],
                "delta": (
                    int(overall[high]["correct"])
                    - int(overall[base]["correct"])
                ),
            }

    payload = {
        "evaluation_protocol": {
            "total_images": 91,
            "calibration_images": 8,
            "held_out_test_images": 83,
            "selected_layer": 17,
            "max_pixels": 262144,
            "visual_token_budget": 256,
            "high_resolution_threshold": 1024,
            "model": "mlx-community/Qwen3-VL-2B-Instruct-4bit",
        },
        "method_scope": {
            "paper_internal_methods": ["rel_att", "grad_att", "pure_grad"],
            "paper_high_resolution_strategy": [
                "rel_att_high",
                "grad_att_high",
                "pure_grad_high",
            ],
            "repository_discrepancy_variant": "pure_grad_repo",
            "implemented_improvement": "rel_att_plus",
        },
        "overall_held_out": overall,
        "by_condition_held_out": by_condition,
        "paired_changes_held_out": paired,
        "synthetic_localization_held_out": {
            variant: dict(values) for variant, values in localization.items()
        },
        "complementarity": {
            "at_least_one_core_method_correct": core_oracle,
            "count": len(sample_ids),
            "rel_att_wrong_grad_att_correct": sum(
                lookup[(sample_id, "rel_att")]["correct"] == "0"
                and lookup[(sample_id, "grad_att")]["correct"] == "1"
                for sample_id in sample_ids
            ),
            "grad_att_wrong_rel_att_correct": sum(
                lookup[(sample_id, "grad_att")]["correct"] == "0"
                and lookup[(sample_id, "rel_att")]["correct"] == "1"
                for sample_id in sample_ids
            ),
        },
        "paper_repository_pure_grad": {
            "different_correctness_on": paper_repo_disagreements,
            "count": len(sample_ids),
            "paper_definition": "pixel gradient norm, 3x3 Gaussian high-pass, 3x3 median",
            "repository_definition": "task/general gradient-norm ratio, 3x3 Gaussian high-pass, 7x7 median",
        },
        "high_resolution_delta": high_resolution_delta,
        "failure_examples": {
            "map_localization_disagreement": "synthetic/002_small_clear",
            "perception_after_correct_localization": "synthetic/004_tiny_clear",
            "single_crop_relation_failure_and_fix": "synthetic/026_multi_region_relation",
            "paper_repository_pure_grad_difference": "synthetic/002_small_clear",
        },
    }
    (ROOT / "results" / "internal_methods_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_condition_csv(
        by_condition,
        ROOT / "results" / "internal_methods_analysis_by_condition.csv",
    )
    output_dir = ROOT / "results" / "figures" / "internal_methods_bw"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_core_accuracy(
        test_rows,
        output_dir / "internal_accuracy_by_condition.png",
    )
    plot_layer_calibration(output_dir)
    make_map_comparison(lookup, output_dir)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
