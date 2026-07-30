from __future__ import annotations

import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .gradient_maps import (
    gradient_weighted_attention_qwen3_vl,
    input_gradient_qwen3_vl,
)
from .rel_att import (
    AUTHOR_QWEN_BBOX_SIZE,
    CropResult,
    _portable_source_path,
    bbox_from_attention_adaptive,
    relative_attention_qwen3_vl,
)


@dataclass(frozen=True)
class HighResolutionBlock:
    row: int
    column: int
    bbox: tuple[int, int, int, int]
    padded_size: tuple[int, int]


@dataclass(frozen=True)
class HighResolutionMapResult:
    importance_map: np.ndarray
    method: str
    split_rows: int
    split_columns: int
    threshold: int
    blocks: tuple[HighResolutionBlock, ...]
    block_grid_shapes: tuple[tuple[int, int], ...]


def split_high_resolution_image(
    image: Image.Image,
    *,
    threshold: int = 1024,
) -> tuple[list[tuple[HighResolutionBlock, Image.Image]], int, int]:
    """Split into non-overlapping, aspect-aware blocks no larger than threshold.

    The paper specifies blocks smaller than 1024 x 1024 with aspect ratios
    close to one.  Starting with the short-axis count and deriving the
    long-axis count from the image aspect ratio keeps the nominal block
    dimensions close to one another.  ``ceil`` is
    essential here: the public implementation's truncation can leave a
    landscape block wider than the stated threshold.  Edge blocks are
    black-padded exactly as PIL's out-of-bounds ``crop`` in the public
    implementation.
    """

    if threshold < 1:
        raise ValueError("threshold must be positive")
    width, height = image.size
    if width >= height:
        rows = max(1, math.ceil(height / threshold))
        columns = max(1, math.ceil(rows * width / height))
    else:
        columns = max(1, math.ceil(width / threshold))
        rows = max(1, math.ceil(columns * height / width))
    block_width = math.ceil(width / columns)
    block_height = math.ceil(height / rows)
    if block_width > threshold or block_height > threshold:
        raise AssertionError("High-resolution split exceeded its threshold")

    blocks: list[tuple[HighResolutionBlock, Image.Image]] = []
    for row in range(rows):
        for column in range(columns):
            left = column * block_width
            top = row * block_height
            right = min(width, left + block_width)
            bottom = min(height, top + block_height)
            valid = image.crop((left, top, right, bottom))
            padded = Image.new("RGB", (block_width, block_height), (0, 0, 0))
            padded.paste(valid, (0, 0))
            blocks.append(
                (
                    HighResolutionBlock(
                        row=row,
                        column=column,
                        bbox=(left, top, right, bottom),
                        padded_size=(block_width, block_height),
                    ),
                    padded,
                )
            )
    return blocks, rows, columns


def stitch_high_resolution_maps(
    block_maps: list[np.ndarray],
    blocks: list[HighResolutionBlock],
    *,
    rows: int,
    columns: int,
) -> np.ndarray:
    """Re-attach block importance maps while removing padded edge cells."""

    if len(block_maps) != len(blocks) or len(blocks) != rows * columns:
        raise ValueError("Block maps and split geometry do not agree")
    trimmed: dict[tuple[int, int], np.ndarray] = {}
    for values, block in zip(block_maps, blocks):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError("Every block importance map must be non-empty and 2D")
        left, top, right, bottom = block.bbox
        padded_width, padded_height = block.padded_size
        keep_width = max(
            1,
            min(
                values.shape[1],
                round(values.shape[1] * (right - left) / padded_width),
            ),
        )
        keep_height = max(
            1,
            min(
                values.shape[0],
                round(values.shape[0] * (bottom - top) / padded_height),
            ),
        )
        trimmed[(block.row, block.column)] = values[:keep_height, :keep_width]

    row_maps = []
    for row in range(rows):
        maps = [trimmed[(row, column)] for column in range(columns)]
        row_heights = {values.shape[0] for values in maps}
        if len(row_heights) != 1:
            common_height = min(row_heights)
            maps = [values[:common_height] for values in maps]
        row_maps.append(np.concatenate(maps, axis=1))
    row_widths = {values.shape[1] for values in row_maps}
    if len(row_widths) != 1:
        common_width = min(row_widths)
        row_maps = [values[:, :common_width] for values in row_maps]
    return np.concatenate(row_maps, axis=0)


def high_resolution_importance_map_qwen3_vl(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    *,
    method: str,
    layer: int,
    threshold: int = 1024,
) -> HighResolutionMapResult:
    """Paper high-resolution stage for all three internal ViCrop methods."""

    valid_methods = {"rel_att", "grad_att", "pure_grad", "pure_grad_repo"}
    if method not in valid_methods:
        raise ValueError(f"Unknown internal ViCrop method: {method}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    split, rows, columns = split_high_resolution_image(
        image,
        threshold=threshold,
    )

    maps: list[np.ndarray] = []
    block_records: list[HighResolutionBlock] = []
    grid_shapes: list[tuple[int, int]] = []
    with tempfile.TemporaryDirectory(prefix="vicrop_high_res_") as temp_name:
        temp_dir = Path(temp_name)
        for index, (block, block_image) in enumerate(split):
            block_path = temp_dir / f"block_{index:04d}.png"
            block_image.save(block_path)
            if method == "rel_att":
                result = relative_attention_qwen3_vl(
                    model,
                    processor,
                    block_path,
                    prompt,
                    layer=layer,
                )
            elif method == "grad_att":
                result = gradient_weighted_attention_qwen3_vl(
                    model,
                    processor,
                    block_path,
                    prompt,
                    layer=layer,
                )
            else:
                repository_variant = method == "pure_grad_repo"
                result = input_gradient_qwen3_vl(
                    model,
                    processor,
                    block_path,
                    prompt,
                    normalize_by_general=repository_variant,
                    median_kernel=7 if repository_variant else 3,
                )
            maps.append(np.asarray(result.attention_map, dtype=np.float64))
            block_records.append(block)
            grid_shapes.append(tuple(int(value) for value in result.grid_shape))

    importance_map = stitch_high_resolution_maps(
        maps,
        block_records,
        rows=rows,
        columns=columns,
    )
    if not np.isfinite(importance_map).all():
        raise FloatingPointError("High-resolution importance map is non-finite")
    return HighResolutionMapResult(
        importance_map=importance_map,
        method=method,
        split_rows=rows,
        split_columns=columns,
        threshold=threshold,
        blocks=tuple(block_records),
        block_grid_shapes=tuple(grid_shapes),
    )


def save_high_resolution_crop(
    image_path: Path,
    output_dir: Path,
    result: HighResolutionMapResult,
    *,
    bbox_size: int = AUTHOR_QWEN_BBOX_SIZE,
) -> tuple[Path, CropResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        crop_result = bbox_from_attention_adaptive(
            result.importance_map,
            image.size,
            bbox_size=bbox_size,
        )
        crop_path = output_dir / "crop.png"
        image.crop(crop_result.bbox).save(crop_path)
    np.save(output_dir / "importance_map.npy", result.importance_map)
    metadata = {
        "source_image": _portable_source_path(image_path),
        "method": f"{result.method}_high",
        "paper_high_resolution_strategy": True,
        "threshold": result.threshold,
        "split_rows": result.split_rows,
        "split_columns": result.split_columns,
        "blocks": [asdict(block) for block in result.blocks],
        "block_grid_shapes": [
            list(shape) for shape in result.block_grid_shapes
        ],
        "combined_grid_shape": list(result.importance_map.shape),
        "bbox_size": bbox_size,
        "crop": asdict(crop_result),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return crop_path, crop_result
