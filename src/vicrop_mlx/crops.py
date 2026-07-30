from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _color_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if color == "red":
        selected_hue = (hue <= 6) | (hue >= 174)
    elif color == "blue":
        selected_hue = (hue >= 105) & (hue <= 125)
    else:
        raise ValueError(f"Unsupported target color: {color}")
    return (selected_hue & (saturation >= 55) & (value >= 95)).astype(np.uint8)


def find_colored_bbox(image: Image.Image, color: str) -> tuple[int, int, int, int]:
    rgb = np.asarray(image.convert("RGB"))
    mask = _color_mask(rgb, color)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = [stats[i] for i in range(1, count) if int(stats[i, 4]) >= 20]
    if not candidates:
        raise ValueError(f"No {color} target region detected")
    x, y, width, height, _ = max(candidates, key=lambda item: int(item[4]))
    return int(x), int(y), int(x + width), int(y + height)


def expanded_square(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    expansion: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    image_width, image_height = image_size
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    side = min(
        max(right - left, bottom - top) * expansion,
        image_width,
        image_height,
    )
    crop_left = round(center_x - side / 2)
    crop_top = round(center_y - side / 2)
    crop_left = min(max(0, crop_left), max(0, round(image_width - side)))
    crop_top = min(max(0, crop_top), max(0, round(image_height - side)))
    crop_right = min(image_width, round(crop_left + side))
    crop_bottom = min(image_height, round(crop_top + side))
    return crop_left, crop_top, crop_right, crop_bottom


def _enhance_and_resize(image: Image.Image, long_side: int = 896) -> Image.Image:
    image = ImageEnhance.Contrast(image.convert("RGB")).enhance(1.12)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.1, percent=140, threshold=2))
    width, height = image.size
    scale = long_side / max(width, height)
    if scale > 1:
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


def save_color_crop(
    image_path: Path,
    color: str,
    output_path: Path,
    expansion: float = 2.5,
) -> Path:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        bbox = find_colored_bbox(image, color)
        crop_box = expanded_square(bbox, image.size, expansion=expansion)
        crop = _enhance_and_resize(image.crop(crop_box))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)
    return output_path


def save_grid_tiles(
    image_path: Path,
    output_dir: Path,
    columns: int = 3,
    rows: int = 2,
    overlap: float = 0.18,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        cell_width = width / columns
        cell_height = height / rows
        for row in range(rows):
            for column in range(columns):
                left = column * cell_width
                top = row * cell_height
                right = (column + 1) * cell_width
                bottom = (row + 1) * cell_height
                left = max(0, round(left - overlap * cell_width))
                top = max(0, round(top - overlap * cell_height))
                right = min(width, round(right + overlap * cell_width))
                bottom = min(height, round(bottom + overlap * cell_height))
                crop = _enhance_and_resize(image.crop((left, top, right, bottom)))
                output = output_dir / f"tile_r{row}_c{column}.png"
                crop.save(output)
                outputs.append(output)
    return outputs


def save_atlas(
    images: list[Path],
    output_path: Path,
    columns: int,
    gap: int = 16,
) -> Path:
    opened = [Image.open(path).convert("RGB") for path in images]
    try:
        cell_width = max(image.width for image in opened)
        cell_height = max(image.height for image in opened)
        rows = (len(opened) + columns - 1) // columns
        canvas = Image.new(
            "RGB",
            (
                columns * cell_width + (columns + 1) * gap,
                rows * cell_height + (rows + 1) * gap,
            ),
            (238, 241, 245),
        )
        for index, image in enumerate(opened):
            row, column = divmod(index, columns)
            left = gap + column * (cell_width + gap) + (cell_width - image.width) // 2
            top = gap + row * (cell_height + gap) + (cell_height - image.height) // 2
            canvas.paste(image, (left, top))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    finally:
        for image in opened:
            image.close()
    return output_path


def prepare_inputs(
    image_path: Path,
    domain: str,
    task: str,
    variant: str,
    crop_root: Path,
    sample_slug: str,
) -> list[Path]:
    if variant == "baseline":
        return [image_path]

    sample_dir = crop_root / variant / sample_slug
    if domain == "imagegen":
        tiles = save_grid_tiles(image_path, sample_dir / "tiles")
        if variant == "single_crop":
            return [tiles[len(tiles) // 2]]
        if variant in {"adaptive", "improved"}:
            return tiles
        raise ValueError(f"Unknown variant: {variant}")

    red = save_color_crop(image_path, "red", sample_dir / "red.png")
    if variant == "single_crop" or task == "code":
        return [red]
    if variant in {"adaptive", "improved"} and task == "relation":
        blue = save_color_crop(image_path, "blue", sample_dir / "blue.png")
        atlas = save_atlas([red, blue], sample_dir / "red_blue_atlas.png", columns=2)
        return [atlas]
    raise ValueError(f"Unknown variant/task combination: {variant}/{task}")
