#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicrop_mlx.dataset import build_samples, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "manifest.csv")
    args = parser.parse_args()
    samples = build_samples(args.root)
    write_manifest(samples, args.root, args.output)
    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()

