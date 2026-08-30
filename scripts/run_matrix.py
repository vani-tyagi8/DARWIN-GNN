from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import yaml

from devign_imbalance.experiment import run

DATASETS = ["devign", "reveal", "bigvul"]
STRATEGIES = ["baseline", "random_oversampling", "random_undersampling",
              "class_weighted", "focal_loss"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled experiment matrix")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2026])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    results = []
    for dataset in args.datasets:
        for strategy in args.strategies:
            for seed in args.seeds:
                config = copy.deepcopy(base)
                config["data"]["dataset"] = dataset
                config["training"]["strategy"] = strategy
                config["seed"] = seed
                metrics_path = (Path(config["output_dir"]) / dataset / strategy /
                                str(seed) / "metrics.json")
                if args.skip_existing and metrics_path.exists():
                    results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                    print(f"Skipping completed run: {dataset} / {strategy} / {seed}")
                    continue
                print(f"\n=== {dataset} / {strategy} / seed={seed} ===", flush=True)
                results.append(run(config))
    output = Path(base["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.json_normalize(results)
    frame.to_csv(output / "all_runs.csv", index=False)
    metric_columns = ["precision", "recall", "f1", "mcc", "pr_auc"]
    summary = frame.groupby(["dataset", "strategy"])[metric_columns].agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    (output / "experiment_manifest.json").write_text(
        json.dumps({"datasets": args.datasets, "strategies": args.strategies, "seeds": args.seeds}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
