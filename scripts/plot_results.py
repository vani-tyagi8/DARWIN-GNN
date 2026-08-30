from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METRICS = ["precision", "recall", "f1", "mcc", "pr_auc"]
ORDER = ["baseline", "random_oversampling", "random_undersampling",
         "class_weighted", "focal_loss"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs/all_runs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/figures"))
    args = parser.parse_args()
    frame = pd.read_csv(args.runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = frame.groupby(["dataset", "strategy"])[METRICS].agg(["mean", "std"])
    summary.to_csv(args.output_dir / "metric_summary.csv")
    for dataset, subset in frame.groupby("dataset"):
        means = subset.groupby("strategy")[METRICS].mean().reindex(ORDER).dropna(how="all")
        errors = subset.groupby("strategy")[METRICS].std().reindex(means.index).fillna(0)
        axes = means.plot.bar(yerr=errors, figsize=(12, 6), capsize=3, ylim=(-0.1, 1.05))
        axes.set_title(f"Devign imbalance strategies - {dataset}")
        axes.set_xlabel("Training strategy")
        axes.set_ylabel("Score (mean +/- standard deviation)")
        axes.legend(loc="lower right", ncol=2)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(args.output_dir / f"{dataset}_strategy_comparison.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()

