from pathlib import Path

from PIL import Image

from vicrop_mlx.crops import find_colored_bbox
from vicrop_mlx.dataset import build_samples
from vicrop_mlx.evaluation import normalize_prediction


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_has_all_supplied_images() -> None:
    samples = build_samples(ROOT)
    assert len(samples) == 91
    assert sum(sample.domain == "synthetic" for sample in samples) == 40
    assert sum(sample.domain == "imagegen" for sample in samples) == 51


def test_known_labels() -> None:
    by_id = {sample.sample_id: sample.answer for sample in build_samples(ROOT)}
    assert by_id["synthetic/000_tiny_clear"] == "4HHV"
    assert by_id["synthetic/026_multi_region_relation"] == "15"
    assert by_id["imagegen/000_rainy_store_Q7M4"] == "Q7M4"


def test_red_locator_finds_tiny_target() -> None:
    with Image.open(ROOT / "synthetic" / "000_tiny_clear.png") as image:
        left, top, right, bottom = find_colored_bbox(image, "red")
    assert (left, top, right, bottom) == (1092, 556, 1129, 575)


def test_normalization() -> None:
    assert normalize_prediction("The code is Q7M4.", "code") == "Q7M4"
    assert normalize_prediction("Q7M4 is the code.", "code") == "Q7M4"
    assert normalize_prediction("11\n", "relation") == "11"
