#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicrop_mlx.crops import find_colored_bbox


VARIANTS = ("baseline", "rel_att", "rel_att_plus")
LABELS = {
    "baseline": "Original only",
    "rel_att": "Strict rel-att",
    "rel_att_plus": "Failure-triggered rel-att+",
}
CONDITION_LABELS = {
    "large_clear": "Large",
    "small_clear": "Small",
    "tiny_clear": "Tiny",
    "small_low_contrast": "Low contrast",
    "multi_region_relation": "Two regions",
    "naturalistic_small_text": "Natural",
}


def font(size: int, bold: bool = False):
    candidates = [
        (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf"
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_bbox(value: str) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    values = tuple(int(item) for item in value.split())
    if len(values) != 4:
        raise ValueError(f"Invalid bbox: {value}")
    return values


def center_inside(
    target: tuple[int, int, int, int],
    crops: list[tuple[int, int, int, int]],
) -> bool:
    center_x = (target[0] + target[2]) / 2
    center_y = (target[1] + target[3]) / 2
    return any(
        left <= center_x <= right and top <= center_y <= bottom
        for left, top, right, bottom in crops
    )


def load_gray(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L").copy()


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.contain(image.convert("L"), size, Image.Resampling.LANCZOS)


def heatmap_gray(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    array = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(array, [2, 98])
    if high <= low:
        normalized = np.zeros_like(array)
    else:
        normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
    image = Image.fromarray(np.uint8(normalized * 255), mode="L")
    return image.resize(size, Image.Resampling.NEAREST)


def bbox_on_image(
    path: Path,
    boxes: list[tuple[tuple[int, int, int, int], str]],
    size: tuple[int, int],
) -> Image.Image:
    with Image.open(path) as source:
        source_gray = source.convert("L")
        original_size = source.size
    image = fit_image(source_gray, size)
    scale = min(size[0] / original_size[0], size[1] / original_size[1])
    offset_x = (size[0] - original_size[0] * scale) / 2
    offset_y = (size[1] - original_size[1] * scale) / 2
    draw = ImageDraw.Draw(image)
    for bbox, style in boxes:
        coords = (
            round(offset_x + bbox[0] * scale),
            round(offset_y + bbox[1] * scale),
            round(offset_x + bbox[2] * scale),
            round(offset_y + bbox[3] * scale),
        )
        if style == "solid":
            draw.rectangle(coords, outline=0, width=5)
        else:
            left, top, right, bottom = coords
            dash = 14
            for x in range(left, right, 2 * dash):
                draw.line((x, top, min(x + dash, right), top), fill=0, width=4)
                draw.line(
                    (x, bottom, min(x + dash, right), bottom),
                    fill=0,
                    width=4,
                )
            for y in range(top, bottom, 2 * dash):
                draw.line((left, y, left, min(y + dash, bottom)), fill=0, width=4)
                draw.line(
                    (right, y, right, min(y + dash, bottom)),
                    fill=0,
                    width=4,
                )
    return image


def panel_figure(
    panels: list[tuple[str, Image.Image]],
    output_path: Path,
    *,
    width: int = 2100,
    panel_height: int = 600,
) -> None:
    margin, gap, title_height = 36, 28, 92
    panel_width = (
        width - 2 * margin - gap * (len(panels) - 1)
    ) // len(panels)
    canvas = Image.new(
        "L",
        (width, panel_height + title_height + 2 * margin),
        255,
    )
    draw = ImageDraw.Draw(canvas)
    title_font = font(28, bold=True)
    for index, (title, image) in enumerate(panels):
        left = margin + index * (panel_width + gap)
        draw.multiline_text(
            (left, margin),
            title,
            font=title_font,
            fill=0,
            spacing=5,
        )
        fitted = fit_image(image, (panel_width, panel_height))
        x = left + (panel_width - fitted.width) // 2
        y = margin + title_height + (panel_height - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        draw.rectangle(
            (left, margin + title_height, left + panel_width, margin + title_height + panel_height),
            outline=160,
            width=2,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def plot_accuracy(
    test_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    conditions = [
        "large_clear",
        "small_clear",
        "tiny_clear",
        "small_low_contrast",
        "multi_region_relation",
        "naturalistic_small_text",
    ]
    width, height = 2200, 920
    left, top, right, bottom = 150, 120, 60, 180
    plot_w, plot_h = width - left - right, height - top - bottom
    canvas = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(canvas)
    axis_font = font(26)
    label_font = font(25)
    title_font = font(34, bold=True)
    draw.text((left, 38), "Held-out exact-match accuracy", font=title_font, fill=0)

    for tick in range(0, 101, 20):
        y = top + plot_h - round(plot_h * tick / 100)
        draw.line((left, y, left + plot_w, y), fill=220, width=2)
        draw.text((70, y - 15), str(tick), font=axis_font, fill=0)
    draw.line((left, top, left, top + plot_h), fill=0, width=3)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=0, width=3)

    group_w = plot_w / len(conditions)
    bar_w = int(group_w * 0.22)
    fills = (225, 150, 65)
    for condition_index, condition in enumerate(conditions):
        center = left + group_w * (condition_index + 0.5)
        for variant_index, variant in enumerate(VARIANTS):
            subset = [
                row
                for row in test_rows
                if row["variant"] == variant and row["condition"] == condition
            ]
            value = 100 * sum(int(row["correct"]) for row in subset) / len(subset)
            x0 = round(center + (variant_index - 1) * (bar_w + 8) - bar_w / 2)
            y0 = top + plot_h - round(plot_h * value / 100)
            draw.rectangle(
                (x0, y0, x0 + bar_w, top + plot_h),
                fill=fills[variant_index],
                outline=0,
                width=2,
            )
            if value > 0:
                draw.text(
                    (x0 + 4, y0 - 31),
                    f"{value:.0f}",
                    font=font(20),
                    fill=0,
                )
        label = CONDITION_LABELS[condition]
        box = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            (center - (box[2] - box[0]) / 2, top + plot_h + 28),
            label,
            font=label_font,
            fill=0,
        )

    legend_x = left + 520
    for index, variant in enumerate(VARIANTS):
        x = legend_x + index * 480
        draw.rectangle((x, 75, x + 44, 105), fill=fills[index], outline=0, width=2)
        draw.text((x + 58, 72), LABELS[variant], font=label_font, fill=0)
    canvas.save(output_dir / "rel_att_accuracy_by_condition.png")


def plot_layer_calibration(output_dir: Path) -> None:
    selection = json.loads(
        (ROOT / "results" / "layer_selection.json").read_text(encoding="utf-8")
    )
    ranking = {int(row["layer"]): row for row in selection["ranking"]}
    layers = sorted(ranking)
    selected = int(selection["selected_layer"])
    width, height = 1800, 720
    left, top, right, bottom = 130, 100, 70, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    canvas = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 28), "Layer calibration (8 fixed images)", font=font(32, True), fill=0)
    for tick in range(0, 101, 25):
        y = top + plot_h - round(plot_h * tick / 100)
        draw.line((left, y, left + plot_w, y), fill=220, width=2)
        draw.text((55, y - 14), str(tick), font=font(23), fill=0)
    draw.line((left, top, left, top + plot_h), fill=0, width=3)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=0, width=3)

    def point(layer: int, key: str) -> tuple[int, int]:
        value = 100 * ranking[layer][key] / ranking[layer]["count"]
        x = left + round(plot_w * layer / max(layers))
        y = top + plot_h - round(plot_h * value / 100)
        return x, y

    for key, fill, dashed in (
        ("correct", 0, False),
        ("center_hits", 115, True),
    ):
        points = [point(layer, key) for layer in layers]
        for first, second in zip(points, points[1:]):
            if dashed:
                for fraction in np.arange(0, 1, 0.16):
                    if int(fraction / 0.16) % 2 == 0:
                        x0 = first[0] + (second[0] - first[0]) * fraction
                        y0 = first[1] + (second[1] - first[1]) * fraction
                        fraction2 = min(1, fraction + 0.08)
                        x1 = first[0] + (second[0] - first[0]) * fraction2
                        y1 = first[1] + (second[1] - first[1]) * fraction2
                        draw.line((x0, y0, x1, y1), fill=fill, width=4)
            else:
                draw.line((*first, *second), fill=fill, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=fill)
    selected_x, _ = point(selected, "correct")
    draw.line((selected_x, top, selected_x, top + plot_h), fill=70, width=3)
    draw.text((selected_x + 10, top + 12), f"selected L{selected}", font=font(24, True), fill=0)
    for layer in range(0, max(layers) + 1, 4):
        x, _ = point(layer, "correct")
        draw.text((x - 10, top + plot_h + 22), str(layer), font=font(22), fill=0)
    draw.text((left + 470, height - 58), "solid: exact match", font=font(24), fill=0)
    draw.text((left + 850, height - 58), "dashed: target-center hit", font=font(24), fill=90)
    canvas.save(output_dir / "rel_att_layer_calibration.png")


def make_example_figures(
    row_lookup: dict[tuple[str, str], dict[str, str]],
    output_dir: Path,
) -> None:
    sample_id = "imagegen/000_rainy_store_Q7M4"
    original_path = ROOT / row_lookup[(sample_id, "baseline")]["image_path"]
    strict = row_lookup[(sample_id, "rel_att")]
    strict_dir = ROOT / "results" / "rel_att" / "rel_att" / sample_id.replace("/", "__")
    panel_figure(
        [
            (
                "Original + selected crop",
                bbox_on_image(
                    original_path,
                    [(parse_bbox(strict["bbox_1"]), "solid")],
                    (660, 600),
                ),
            ),
            (
                "Layer-17 relative attention",
                heatmap_gray(
                    np.load(strict_dir / "relative_attention.npy"),
                    (660, 600),
                ),
            ),
            (
                f"Crop\noriginal={row_lookup[(sample_id, 'baseline')]['prediction']}  "
                f"rel-att={strict['prediction']}",
                load_gray(strict_dir / "crop.png"),
            ),
        ],
        output_dir / "rel_att_natural_success.png",
    )

    sample_id = "synthetic/002_small_clear"
    original_path = ROOT / row_lookup[(sample_id, "baseline")]["image_path"]
    strict = row_lookup[(sample_id, "rel_att")]
    strict_dir = ROOT / "results" / "rel_att" / "rel_att" / sample_id.replace("/", "__")
    with Image.open(original_path) as source:
        target = find_colored_bbox(source.convert("RGB"), "red")
    panel_figure(
        [
            (
                "Dashed: target / Solid: wrong crop",
                bbox_on_image(
                    original_path,
                    [
                        (target, "dashed"),
                        (parse_bbox(strict["bbox_1"]), "solid"),
                    ],
                    (660, 600),
                ),
            ),
            (
                "Relative attention",
                heatmap_gray(
                    np.load(strict_dir / "relative_attention.npy"),
                    (660, 600),
                ),
            ),
            (
                f"Empty crop\nGT={strict['answer']}  output={strict['prediction']}",
                load_gray(strict_dir / "crop.png"),
            ),
        ],
        output_dir / "rel_att_failure_localization.png",
    )

    sample_id = "synthetic/004_tiny_clear"
    original_path = ROOT / row_lookup[(sample_id, "baseline")]["image_path"]
    strict = row_lookup[(sample_id, "rel_att")]
    strict_dir = ROOT / "results" / "rel_att" / "rel_att" / sample_id.replace("/", "__")
    with Image.open(original_path) as source:
        source_gray = source.convert("L")
        target = find_colored_bbox(source.convert("RGB"), "red")
        target_detail = source_gray.crop(target)
    panel_figure(
        [
            (
                "Correctly localized crop",
                bbox_on_image(
                    original_path,
                    [(parse_bbox(strict["bbox_1"]), "solid")],
                    (660, 600),
                ),
            ),
            ("224 px context crop", load_gray(strict_dir / "crop.png")),
            (
                f"Target detail\nGT={strict['answer']}  output={strict['prediction']}",
                target_detail,
            ),
        ],
        output_dir / "rel_att_failure_ocr.png",
    )

    sample_id = "synthetic/026_multi_region_relation"
    original_path = ROOT / row_lookup[(sample_id, "baseline")]["image_path"]
    strict = row_lookup[(sample_id, "rel_att")]
    plus = row_lookup[(sample_id, "rel_att_plus")]
    strict_dir = ROOT / "results" / "rel_att" / "rel_att" / sample_id.replace("/", "__")
    plus_dir = ROOT / "results" / "rel_att" / "rel_att_plus" / sample_id.replace("/", "__")
    crop_1 = fit_image(load_gray(plus_dir / "crop_1.png"), (310, 550))
    crop_2 = fit_image(load_gray(plus_dir / "crop_2.png"), (310, 550))
    both = Image.new("L", (640, max(crop_1.height, crop_2.height)), 255)
    both.paste(crop_1, (0, 0))
    both.paste(crop_2, (330, 0))
    panel_figure(
        [
            (
                "Solid: crop 1 / Dashed: crop 2",
                bbox_on_image(
                    original_path,
                    [
                        (parse_bbox(plus["bbox_1"]), "solid"),
                        (parse_bbox(plus["bbox_2"]), "dashed"),
                    ],
                    (660, 600),
                ),
            ),
            (
                f"Strict: one region\noutput={strict['prediction']}",
                load_gray(strict_dir / "crop.png"),
            ),
            (
                f"rel-att+: both regions\noutput={plus['prediction']}",
                both,
            ),
        ],
        output_dir / "rel_att_multi_region_fix.png",
    )


def main() -> None:
    rows = read_rows(ROOT / "results" / "rel_att_predictions.csv")
    test_rows = [row for row in rows if row["is_calibration"] == "0"]
    row_lookup = {
        (row["sample_id"], row["variant"]): row for row in rows
    }
    sample_ids = sorted(
        {row["sample_id"] for row in test_rows if row["variant"] == "baseline"}
    )

    overall = {}
    by_condition = []
    for variant in VARIANTS:
        subset = [row for row in test_rows if row["variant"] == variant]
        correct = sum(int(row["correct"]) for row in subset)
        overall[variant] = {
            "correct": correct,
            "count": len(subset),
            "accuracy": correct / len(subset),
            "mean_elapsed_s": statistics.mean(
                float(row["elapsed_s"]) for row in subset
            ),
        }
        for condition in sorted({row["condition"] for row in subset}):
            condition_rows = [
                row for row in subset if row["condition"] == condition
            ]
            by_condition.append(
                {
                    "variant": variant,
                    "condition": condition,
                    "correct": sum(
                        int(row["correct"]) for row in condition_rows
                    ),
                    "count": len(condition_rows),
                }
            )

    paired = {}
    for variant in ("rel_att", "rel_att_plus"):
        paired[variant] = {
            "baseline_error_corrected": sum(
                row_lookup[(sample_id, "baseline")]["correct"] == "0"
                and row_lookup[(sample_id, variant)]["correct"] == "1"
                for sample_id in sample_ids
            ),
            "baseline_correct_harmed": sum(
                row_lookup[(sample_id, "baseline")]["correct"] == "1"
                and row_lookup[(sample_id, variant)]["correct"] == "0"
                for sample_id in sample_ids
            ),
        }
    paired["plus_vs_strict"] = {
        "strict_error_corrected": sum(
            row_lookup[(sample_id, "rel_att")]["correct"] == "0"
            and row_lookup[(sample_id, "rel_att_plus")]["correct"] == "1"
            for sample_id in sample_ids
        ),
        "strict_correct_harmed": sum(
            row_lookup[(sample_id, "rel_att")]["correct"] == "1"
            and row_lookup[(sample_id, "rel_att_plus")]["correct"] == "0"
            for sample_id in sample_ids
        ),
    }

    localization = defaultdict(lambda: defaultdict(int))
    for sample_id in sample_ids:
        baseline = row_lookup[(sample_id, "baseline")]
        if baseline["domain"] != "synthetic":
            continue
        image_path = ROOT / baseline["image_path"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            targets = [find_colored_bbox(image, "red")]
            if baseline["task"] == "relation":
                targets.append(find_colored_bbox(image, "blue"))
        for variant in ("rel_att", "rel_att_plus"):
            row = row_lookup[(sample_id, variant)]
            crops = [
                bbox
                for bbox in (
                    parse_bbox(row["bbox_1"]),
                    parse_bbox(row["bbox_2"]),
                )
                if bbox is not None
            ]
            hit_count = sum(center_inside(target, crops) for target in targets)
            localization[variant]["target_centers"] += len(targets)
            localization[variant]["target_center_hits"] += hit_count
            localization[variant]["samples"] += 1
            localization[variant]["all_targets_hit"] += int(
                hit_count == len(targets)
            )
            if baseline["task"] == "code" and row["correct"] == "0":
                if hit_count == 1:
                    localization[variant]["wrong_after_localization"] += 1
                else:
                    localization[variant]["wrong_from_localization"] += 1

    payload = {
        "evaluation_protocol": {
            "total_images": 91,
            "calibration_images": 8,
            "held_out_test_images": 83,
            "selected_layer": 17,
            "ensemble_layers": [17, 21, 20],
            "max_pixels": 262144,
            "visual_token_budget": 256,
        },
        "overall_held_out": overall,
        "by_condition_held_out": by_condition,
        "paired_changes_held_out": paired,
        "synthetic_localization_held_out": {
            variant: dict(values) for variant, values in localization.items()
        },
        "failure_examples": {
            "localization_failure": "synthetic/002_small_clear",
            "ocr_after_correct_localization": "synthetic/004_tiny_clear",
            "multi_region_fix": "synthetic/026_multi_region_relation",
            "natural_success": "imagegen/000_rainy_store_Q7M4",
        },
    }
    (ROOT / "results" / "rel_att_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    output_dir = ROOT / "results" / "figures" / "rel_att_bw"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy(test_rows, output_dir)
    plot_layer_calibration(output_dir)
    make_example_figures(row_lookup, output_dir)
    print(json.dumps(payload["overall_held_out"], indent=2))


if __name__ == "__main__":
    main()
