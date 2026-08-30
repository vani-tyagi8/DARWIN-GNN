"""Convert the public epicosy/Devign 994-function AST-only sample.

This converter exists strictly for pipeline smoke testing. Its output metadata
marks it as AST-only and must not be reported as the final composite-graph study.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import types
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_legacy_frame(path: Path) -> pd.DataFrame:
    # Pickles were produced with an older Pandas module path.
    legacy = types.ModuleType("pandas.core.indexes.numeric")
    legacy.Int64Index = pd.Index
    legacy.UInt64Index = pd.Index
    legacy.Float64Index = pd.Index
    sys.modules[legacy.__name__] = legacy
    with path.open("rb") as stream:
        return pickle.load(stream)  # noqa: S301 - trusted public repository artifact


def convert_row(row, sample_id: str) -> dict:
    # Old PyG Data instances retain tensors directly in __dict__.
    values = row["input"].__dict__
    features = values["x"].tolist()
    edge_index = values["edge_index"]
    edges = [[int(source), 0, int(target)] for source, target in edge_index.t().tolist()]
    return {
        "id": sample_id,
        "node_features": features,
        "graph": edges,
        "target": int(row["target"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/devign_smoke"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frames = [load_legacy_frame(args.input_dir / f"{index}_cpg_input.pkl") for index in (0, 1)]
    frame = pd.concat(frames, ignore_index=True)
    if args.limit:
        frame, _ = train_test_split(frame, train_size=args.limit, stratify=frame["target"],
                                    random_state=args.seed)
    indices = list(range(len(frame)))
    train_ids, held_ids = train_test_split(indices, test_size=0.2, stratify=frame["target"],
                                           random_state=args.seed)
    valid_ids, test_ids = train_test_split(held_ids, test_size=0.5,
                                           stratify=frame.iloc[held_ids]["target"],
                                           random_state=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, selected in (("train", train_ids), ("valid", valid_ids), ("test", test_ids)):
        records = [convert_row(frame.iloc[index], f"epicosy-{index}") for index in selected]
        (args.output_dir / f"{name}.json").write_text(json.dumps(records), encoding="utf-8")
    metadata = {
        "purpose": "smoke_test_only",
        "source": "https://github.com/epicosy/devign",
        "representation": "AST-only",
        "feature_dim": 101,
        "edge_types": {"0": "AST"},
        "seed": args.seed,
        "total": len(frame),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

