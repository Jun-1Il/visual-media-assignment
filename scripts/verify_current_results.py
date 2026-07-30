#!/usr/bin/env python3
"""Verify that results/ contains one complete, internally consistent experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

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
MAP_VARIANTS = (
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
TOP_LEVEL = {
    "README.md",
    "attention_reconstruction_validation.json",
    "calibration_crops",
    "figures",
    "gradient_internal_validation.json",
    "internal_methods_analysis.json",
    "internal_methods_analysis_by_condition.csv",
    "internal_methods_metrics_by_condition.csv",
    "internal_methods_predictions.csv",
    "internal_methods_summary.json",
    "layer_calibration.csv",
    "layer_selection.json",
    "rel_att",
}
GITHUB_TOP_LEVEL = TOP_LEVEL - {"calibration_crops"}
GITHUB_MAP_SAMPLES = {
    "rel_att": {
        "synthetic__002_small_clear",
        "synthetic__004_tiny_clear",
        "synthetic__026_multi_region_relation",
    },
    "grad_att": {"synthetic__002_small_clear"},
    "pure_grad": {"synthetic__002_small_clear"},
    "pure_grad_repo": {"synthetic__002_small_clear"},
    "rel_att_plus": {"synthetic__026_multi_region_relation"},
}
FIGURES = {
    "failure_perception_after_localization.png",
    "failure_relation_and_fix.png",
    "internal_accuracy_by_condition.png",
    "internal_method_map_comparison.png",
    "pure_grad_paper_repository_difference.png",
    "rel_att_layer_calibration.png",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="verify the intentionally reduced GitHub bundle",
    )
    args = parser.parse_args()
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    actual_top = {path.name for path in RESULTS.iterdir()}
    expected_top = GITHUB_TOP_LEVEL if args.bundle else TOP_LEVEL
    check(
        actual_top == expected_top,
        "results/ top-level mismatch: "
        f"missing={sorted(expected_top - actual_top)}, "
        f"unexpected={sorted(actual_top - expected_top)}",
    )

    predictions = read_csv(RESULTS / "internal_methods_predictions.csv")
    check(len(predictions) == 1001, f"prediction rows: {len(predictions)} != 1001")
    variant_counts = Counter(row["variant"] for row in predictions)
    check(
        variant_counts == Counter({variant: 91 for variant in VARIANTS}),
        f"prediction variant counts mismatch: {dict(variant_counts)}",
    )

    samples_by_variant: dict[str, set[str]] = defaultdict(set)
    for row in predictions:
        samples_by_variant[row["variant"]].add(row["sample_id"])
    baseline_samples = samples_by_variant["baseline"]
    check(len(baseline_samples) == 91, f"unique samples: {len(baseline_samples)} != 91")
    for variant in VARIANTS:
        check(
            samples_by_variant[variant] == baseline_samples,
            f"sample set differs for {variant}",
        )

    calibration_ids = {
        row["sample_id"] for row in predictions if row["is_calibration"] == "1"
    }
    held_out_ids = baseline_samples - calibration_ids
    check(len(calibration_ids) == 8, f"calibration samples: {len(calibration_ids)} != 8")
    check(len(held_out_ids) == 83, f"held-out samples: {len(held_out_ids)} != 83")

    summary = read_json(RESULTS / "internal_methods_summary.json")
    analysis = read_json(RESULTS / "internal_methods_analysis.json")
    for variant in VARIANTS:
        all_rows = [row for row in predictions if row["variant"] == variant]
        test_rows = [
            row
            for row in all_rows
            if row["sample_id"] in held_out_ids
        ]
        all_correct = sum(int(row["correct"]) for row in all_rows)
        test_correct = sum(int(row["correct"]) for row in test_rows)
        summary_all = summary["all_91"][variant]
        summary_test = summary["held_out_test"][variant]
        analysis_test = analysis["overall_held_out"][variant]
        check(
            (summary_all["correct"], summary_all["count"])
            == (all_correct, 91),
            f"all-91 summary mismatch for {variant}",
        )
        check(
            (summary_test["correct"], summary_test["count"])
            == (test_correct, 83),
            f"held-out summary mismatch for {variant}",
        )
        check(
            (analysis_test["correct"], analysis_test["count"])
            == (test_correct, 83),
            f"held-out analysis mismatch for {variant}",
        )

    for filename in (
        "internal_methods_metrics_by_condition.csv",
        "internal_methods_analysis_by_condition.csv",
    ):
        rows = read_csv(RESULTS / filename)
        check(len(rows) == 66, f"{filename}: {len(rows)} rows != 66")
        check(
            Counter(row["variant"] for row in rows)
            == Counter({variant: 6 for variant in VARIANTS}),
            f"{filename}: variant/condition coverage mismatch",
        )

    selection = read_json(RESULTS / "layer_selection.json")
    check(selection["selected_layer"] == 17, "selected layer is not 17")
    check(selection["ensemble_layers"] == [17, 21, 20], "ensemble layers differ")
    check(
        set(selection["calibration_sample_ids"]) == calibration_ids,
        "calibration sample IDs differ between predictions and layer selection",
    )

    calibration_rows = read_csv(RESULTS / "layer_calibration.csv")
    check(len(calibration_rows) == 224, "layer calibration is not 8 x 28 rows")
    calibration_layers: dict[str, set[int]] = defaultdict(set)
    for row in calibration_rows:
        calibration_layers[row["sample_id"]].add(int(row["layer"]))
    check(
        set(calibration_layers) == calibration_ids,
        "layer calibration sample IDs differ",
    )
    for sample_id in calibration_ids:
        check(
            calibration_layers[sample_id] == set(range(28)),
            f"layer coverage differs for {sample_id}",
        )

    if not args.bundle:
        crop_root = RESULTS / "calibration_crops"
        crop_dirs = {path.name for path in crop_root.iterdir() if path.is_dir()}
        expected_crop_dirs = {
            sample_id.replace("/", "__") for sample_id in calibration_ids
        }
        check(
            crop_dirs == expected_crop_dirs,
            "calibration crop sample directories differ",
        )
        expected_crop_names = {f"layer_{layer:02d}.png" for layer in range(28)}
        for directory in crop_dirs:
            names = {
                path.name
                for path in (crop_root / directory).iterdir()
                if path.is_file()
            }
            check(
                names == expected_crop_names,
                f"calibration crops incomplete: {directory}",
            )

    map_root = RESULTS / "rel_att"
    actual_map_variants = {
        path.name for path in map_root.iterdir() if path.is_dir()
    }
    expected_sample_dirs = {sample_id.replace("/", "__") for sample_id in baseline_samples}
    if args.bundle:
        expected_maps = GITHUB_MAP_SAMPLES
    else:
        expected_maps = {
            variant: expected_sample_dirs for variant in MAP_VARIANTS
        }
    check(
        actual_map_variants == set(expected_maps),
        "raw-map variant directories differ: "
        f"{sorted(actual_map_variants)}",
    )
    for variant, expected_variant_samples in expected_maps.items():
        variant_root = map_root / variant
        sample_dirs = {
            path.name for path in variant_root.iterdir() if path.is_dir()
        }
        check(
            sample_dirs == expected_variant_samples,
            f"raw-map samples differ for {variant}",
        )
        for sample_dir in sample_dirs:
            directory = variant_root / sample_dir
            check(
                any(directory.glob("crop*.png")),
                f"missing crop: {variant}/{sample_dir}",
            )
            check(
                (directory / "metadata.json").is_file(),
                f"missing metadata: {variant}/{sample_dir}",
            )
            check(
                any(directory.glob("*.npy")),
                f"missing map array: {variant}/{sample_dir}",
            )

    for filename in (
        "attention_reconstruction_validation.json",
        "gradient_internal_validation.json",
    ):
        payload = read_json(RESULTS / filename)
        check(payload.get("passed") is True, f"{filename}: passed is not true")

    figure_root = RESULTS / "figures" / "internal_methods_bw"
    figure_names = {
        path.name for path in figure_root.iterdir() if path.is_file()
    }
    check(figure_names == FIGURES, f"current figure set differs: {sorted(figure_names)}")
    figure_dirs = {
        path.name for path in (RESULTS / "figures").iterdir() if path.is_dir()
    }
    check(
        figure_dirs == {"internal_methods_bw"},
        f"unexpected figure directories: {sorted(figure_dirs)}",
    )
    for filename in figure_names:
        with Image.open(figure_root / filename) as image:
            if image.mode == "L":
                continue
            rgb = image.convert("RGB")
            red, green, blue = rgb.split()
            check(
                ImageChops.difference(red, green).getbbox() is None
                and ImageChops.difference(red, blue).getbbox() is None,
                f"figure is not grayscale: {filename}",
            )

    if errors:
        print("CURRENT RESULTS VERIFICATION: FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("CURRENT RESULTS VERIFICATION: PASSED")
    print("- one canonical experiment")
    print("- 91 images, 8 calibration, 83 held-out")
    print("- 11 variants, 1001 prediction rows")
    if args.bundle:
        print("- 7 representative raw-map/sample pairs")
        print("- layer calibration table: 8 images x 28 layers")
    else:
        print("- 10 raw-map variants x 91 images")
        print("- 8 calibration images x 28 layers and crops")
    print("- 2 numerical validations passed")
    print("- 6 grayscale report figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
