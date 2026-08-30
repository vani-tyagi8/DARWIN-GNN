from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .experiment import run
from .sampling import STRATEGIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["devign", "reveal", "bigvul", "devign_smoke"])
    parser.add_argument("--strategy", choices=sorted(STRATEGIES))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.dataset:
        config["data"]["dataset"] = args.dataset
    if args.strategy:
        config["training"]["strategy"] = args.strategy
    if args.seed is not None:
        config["seed"] = args.seed
    print(json.dumps(run(config), indent=2))


if __name__ == "__main__":
    main()
