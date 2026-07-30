from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CODE_QUESTION = (
    "Read the exact four-character code inside the red outlined box. "
    "Answer with only the four characters."
)
NATURAL_QUESTION = (
    "Read the exact four-character alphanumeric code printed in the scene. "
    "Answer with only the four characters."
)
RELATION_QUESTION = (
    "Add the digit inside the red outlined box to the digit inside the blue "
    "outlined box. Answer with only the sum as an integer."
)

# These labels were read from the deliberately rendered synthetic targets and
# visually checked on enlarged crops. The tiny/small pair shares the same code.
PAIRED_CODE_ANSWERS = {
    "000": "4HHV",
    "001": "HCKX",
    "002": "LZH4",
    "003": "FAVH",
    "004": "JFNH",
    "005": "PTSN",
    "006": "TZUC",
    "007": "YAKX",
}
LARGE_CODE_ANSWERS = {
    "008": "ZBVG",
    "009": "WPH9",
    "010": "GNZ8",
    "011": "C54T",
    "012": "EMWE",
    "013": "QCHL",
    "014": "6CJS",
    "015": "T9VX",
}
LOW_CONTRAST_CODE_ANSWERS = {
    "016": "9RMY",
    "017": "D5LF",
    "018": "XXYM",
    "019": "PBSC",
    "020": "UYV3",
    "021": "3CA4",
    "022": "RENR",
    "023": "6XFJ",
}
# Ground truth is red digit + blue digit. Both operands were independently
# checked on enlarged target crops.
RELATION_ANSWERS = {
    "024": "9",   # 4 + 5
    "025": "11",  # 8 + 3
    "026": "15",  # 6 + 9
    "027": "10",  # 5 + 5
    "028": "12",  # 3 + 9
    "029": "12",  # 4 + 8
    "030": "9",   # 5 + 4
    "031": "12",  # 3 + 9
}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    domain: str
    condition: str
    task: str
    question: str
    answer: str
    pair_id: str = ""

    def as_manifest_row(self, root: Path) -> dict[str, str]:
        row = asdict(self)
        row["image_path"] = self.image_path.relative_to(root).as_posix()
        return row


def _synthetic_answer(index: str, condition: str) -> tuple[str, str]:
    if condition in {"tiny_clear", "small_clear"}:
        return PAIRED_CODE_ANSWERS[index], "code"
    if condition == "large_clear":
        return LARGE_CODE_ANSWERS[index], "code"
    if condition == "small_low_contrast":
        return LOW_CONTRAST_CODE_ANSWERS[index], "code"
    if condition == "multi_region_relation":
        return RELATION_ANSWERS[index], "relation"
    raise ValueError(f"Unknown synthetic condition: {condition}")


def build_samples(root: Path) -> list[Sample]:
    root = root.resolve()
    samples: list[Sample] = []

    for path in sorted((root / "synthetic").glob("*.png")):
        match = re.fullmatch(r"(\d{3})_(.+)", path.stem)
        if not match:
            raise ValueError(f"Unexpected synthetic filename: {path.name}")
        index, condition = match.groups()
        answer, task = _synthetic_answer(index, condition)
        samples.append(
            Sample(
                sample_id=f"synthetic/{path.stem}",
                image_path=path.resolve(),
                domain="synthetic",
                condition=condition,
                task=task,
                question=RELATION_QUESTION if task == "relation" else CODE_QUESTION,
                answer=answer,
                pair_id=index if condition in {"tiny_clear", "small_clear"} else "",
            )
        )

    for path in sorted((root / "imagegen").glob("*.png")):
        answer = path.stem.rsplit("_", 1)[-1].upper()
        if not re.fullmatch(r"[A-Z0-9]{4}", answer):
            raise ValueError(f"Filename does not end in a four-character label: {path.name}")
        samples.append(
            Sample(
                sample_id=f"imagegen/{path.stem}",
                image_path=path.resolve(),
                domain="imagegen",
                condition="naturalistic_small_text",
                task="code",
                question=NATURAL_QUESTION,
                answer=answer,
            )
        )

    return samples


def write_manifest(samples: Iterable[Sample], root: Path, output_path: Path) -> None:
    rows = [sample.as_manifest_row(root.resolve()) for sample in samples]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_id",
                "image_path",
                "domain",
                "condition",
                "task",
                "question",
                "answer",
                "pair_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

