from __future__ import annotations

import argparse
import json
from pathlib import Path

from devign_imbalance.data import class_distribution, load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    report = {}
    for split in ("train", "valid", "test"):
        report[split] = class_distribution(load_split(args.dataset_dir, split))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
