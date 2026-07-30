from __future__ import annotations

import numpy as np
from PIL import Image

from vicrop_mlx.high_resolution import (
    split_high_resolution_image,
    stitch_high_resolution_maps,
)
from vicrop_mlx.rel_att import (
    GENERAL_PROMPT,
    bbox_from_attention_adaptive,
    topk_bboxes_from_attention,
)
from vicrop_mlx.gradient_maps import unpatchify_qwen3_vl_pixel_gradient


def test_general_prompt_matches_official_qwen_branch() -> None:
    assert GENERAL_PROMPT == (
        "Write a general description of the image. "
        "Answer the question using a single word or phrase."
    )


def test_adaptive_bbox_tracks_attention_peak() -> None:
    attention = np.zeros((10, 16), dtype=np.float64)
    attention[2:4, 12:14] = 10.0
    result = bbox_from_attention_adaptive(
        attention,
        (1600, 1000),
        bbox_size=224,
    )
    left, top, right, bottom = result.bbox
    assert left <= 1300 <= right
    assert top <= 300 <= bottom
    assert right - left == bottom - top
    assert result.ratio in {1.0, 1.2, 1.4, 1.6, 1.8, 2.0}


def test_adaptive_bbox_returns_full_image_when_first_window_covers_map() -> None:
    attention = np.ones((2, 2), dtype=np.float64)
    result = bbox_from_attention_adaptive(
        attention,
        (200, 200),
        bbox_size=224,
    )
    assert result.bbox == (0, 0, 200, 200)


def test_topk_bbox_suppression_returns_distinct_regions() -> None:
    attention = np.zeros((10, 16), dtype=np.float64)
    attention[2, 2] = 20.0
    attention[7, 13] = 18.0
    results = topk_bboxes_from_attention(
        attention,
        (1600, 1000),
        k=2,
        bbox_size=224,
    )
    first_center = (
        (results[0].bbox[0] + results[0].bbox[2]) / 2,
        (results[0].bbox[1] + results[0].bbox[3]) / 2,
    )
    second_center = (
        (results[1].bbox[0] + results[1].bbox[2]) / 2,
        (results[1].bbox[1] + results[1].bbox[3]) / 2,
    )
    assert abs(first_center[0] - second_center[0]) > 400
    assert abs(first_center[1] - second_center[1]) > 300


def test_qwen_pixel_gradient_unpatchify_reverses_processor_order() -> None:
    patch_size = 2
    temporal_patch_size = 2
    merge_size = 2
    grid_h, grid_w = 4, 6
    image = np.arange(
        3 * grid_h * patch_size * grid_w * patch_size,
        dtype=np.float32,
    ).reshape(3, grid_h * patch_size, grid_w * patch_size)
    patches = np.repeat(
        image[None, None, ...],
        temporal_patch_size,
        axis=1,
    )
    patches = patches.reshape(
        1,
        1,
        temporal_patch_size,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    flattened = patches.transpose(
        0,
        1,
        4,
        7,
        5,
        8,
        3,
        2,
        6,
        9,
    ).reshape(
        grid_h * grid_w,
        3 * temporal_patch_size * patch_size * patch_size,
    )
    restored = unpatchify_qwen3_vl_pixel_gradient(
        flattened,
        (1, grid_h, grid_w),
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        merge_size=merge_size,
    )
    np.testing.assert_array_equal(restored, image * temporal_patch_size)


def test_paper_high_resolution_split_respects_both_axes() -> None:
    image = Image.new("RGB", (2050, 1100), "white")
    split, rows, columns = split_high_resolution_image(image, threshold=1024)
    assert (rows, columns) == (2, 4)
    assert len(split) == 8
    for block, padded in split:
        assert padded.width <= 1024
        assert padded.height <= 1024
        assert block.padded_size == padded.size


def test_paper_high_resolution_split_tracks_extreme_aspect_ratio() -> None:
    landscape = Image.new("RGB", (4096, 256), "white")
    split, rows, columns = split_high_resolution_image(
        landscape,
        threshold=1024,
    )
    assert (rows, columns) == (1, 16)
    for _, block in split:
        assert block.size == (256, 256)

    portrait = Image.new("RGB", (256, 4096), "white")
    split, rows, columns = split_high_resolution_image(
        portrait,
        threshold=1024,
    )
    assert (rows, columns) == (16, 1)
    for _, block in split:
        assert block.size == (256, 256)


def test_high_resolution_map_stitch_removes_edge_padding() -> None:
    image = Image.new("RGB", (2050, 1100), "white")
    split, rows, columns = split_high_resolution_image(image, threshold=1024)
    blocks = [block for block, _ in split]
    maps = [
        np.full((10, 12), index + 1, dtype=np.float64)
        for index in range(len(blocks))
    ]
    combined = stitch_high_resolution_maps(
        maps,
        blocks,
        rows=rows,
        columns=columns,
    )
    assert combined.ndim == 2
    assert combined.shape[0] <= 20
    assert combined.shape[1] <= 48
    assert set(np.unique(combined)) == {1, 2, 3, 4, 5, 6, 7, 8}
