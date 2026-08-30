from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


def _pick(record: dict, *names: str):
    for name in names:
        if name in record:
            return record[name]
    raise KeyError(f"none of {names} found in graph record")


def record_to_graph(record: dict, index: int, id_prefix: str) -> Data:
    features = _pick(record, "node_features", "features")
    edges = _pick(record, "graph", "structure")
    raw_label = _pick(record, "target", "targets", "label")
    while isinstance(raw_label, list):
        raw_label = raw_label[0]
    edge_index = torch.tensor([[e[0] for e in edges], [e[2] for e in edges]], dtype=torch.long)
    edge_type = torch.tensor([e[1] for e in edges], dtype=torch.long)
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
    graph = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor(float(raw_label), dtype=torch.float32),
    )
    graph.sample_id = str(record.get("id", f"{id_prefix}-{index}"))
    return graph


def load_graphs(path: str | Path) -> list[Data]:
    """Load either saikat107 or conventional Devign JSON graph records."""
    with Path(path).open(encoding="utf-8") as stream:
        records = json.load(stream)
    graphs = []
    prefix = Path(path).stem
    for index, record in enumerate(records):
        if not _pick(record, "node_features", "features"):
            continue
        graphs.append(record_to_graph(record, index, prefix))
    return graphs


class IndexedJsonlGraphDataset(Dataset):
    """Random-access, disk-backed graph dataset for multi-gigabyte corpora."""

    def __init__(self, jsonl_path: str | Path, index_path: str | Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        metadata = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.offsets = metadata["offsets"]
        self.labels = metadata["labels"]
        self.id_prefix = metadata["id_prefix"]
        self.feature_dim = int(metadata["feature_dim"])
        self.num_edge_types = int(metadata["num_edge_types"])
        self._stream = None
        self._stream_pid = None

    def __len__(self) -> int:
        return len(self.offsets)

    def _handle(self):
        current_pid = os.getpid()
        if (self._stream is None or self._stream.closed or
                self._stream_pid != current_pid):
            if self._stream is not None and not self._stream.closed:
                self._stream.close()
            self._stream = self.jsonl_path.open("rb")
            self._stream_pid = current_pid
        return self._stream

    def __getitem__(self, index: int) -> Data:
        stream = self._handle()
        stream.seek(self.offsets[index])
        record = json.loads(stream.readline())
        return record_to_graph(record, index, self.id_prefix)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_stream"] = None
        state["_stream_pid"] = None
        return state

    def __del__(self):
        if self._stream is not None:
            self._stream.close()


def load_split(root: str | Path, split: str):
    root = Path(root)
    jsonl = root / f"{split}.jsonl"
    index = root / f"{split}.index.json"
    if jsonl.exists() and index.exists():
        return IndexedJsonlGraphDataset(jsonl, index)
    return load_graphs(root / f"{split}.json")


def validate_graphs(graphs: Sequence[Data], expected_feature_dim: int,
                    expected_edge_types: int) -> None:
    if not graphs:
        raise ValueError("dataset split contains no usable graphs")
    for graph in graphs:
        if graph.x.ndim != 2 or graph.x.size(1) != expected_feature_dim:
            raise ValueError(f"{graph.sample_id}: expected feature width {expected_feature_dim}")
        if graph.edge_index.numel():
            if int(graph.edge_index.min()) < 0 or int(graph.edge_index.max()) >= graph.num_nodes:
                raise ValueError(f"{graph.sample_id}: edge references an invalid node")
            if int(graph.edge_type.min()) < 0 or int(graph.edge_type.max()) >= expected_edge_types:
                raise ValueError(f"{graph.sample_id}: edge type is outside configured vocabulary")
        if float(graph.y.item()) not in {0.0, 1.0}:
            raise ValueError(f"{graph.sample_id}: label must be binary")


def validate_split_bundle(train: Sequence[Data], valid: Sequence[Data], test: Sequence[Data]) -> None:
    if all(isinstance(split, IndexedJsonlGraphDataset) for split in (train, valid, test)):
        prefixes = {train.id_prefix, valid.id_prefix, test.id_prefix}
        if len(prefixes) != 3:
            raise ValueError("disk-backed splits must use distinct ID prefixes")
        for name, split in zip(("train", "valid", "test"), (train, valid, test)):
            if set(split.labels) != {0, 1}:
                raise ValueError(f"{name} split must contain both classes")
        return
    split_ids = [{str(graph.sample_id) for graph in graphs} for graphs in (train, valid, test)]
    if any(len(ids) != len(graphs) for ids, graphs in zip(split_ids, (train, valid, test))):
        raise ValueError("duplicate sample ID within a split")
    if split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2] or split_ids[1] & split_ids[2]:
        raise ValueError("sample ID occurs in more than one split")
    for name, graphs in zip(("train", "valid", "test"), (train, valid, test)):
        labels = {int(graph.y.item()) for graph in graphs}
        if labels != {0, 1}:
            raise ValueError(f"{name} split must contain both classes")


def labels_for(graphs: Sequence[Data]) -> list[int]:
    if hasattr(graphs, "labels"):
        return list(graphs.labels)
    return [int(graph.y.item()) for graph in graphs]


def class_distribution(graphs: Sequence[Data] | None = None,
                       labels: Sequence[int] | None = None) -> dict[str, float | int]:
    values = list(labels) if labels is not None else labels_for(graphs)
    positive = sum(values)
    negative = len(values) - positive
    return {
        "total": len(values),
        "vulnerable": positive,
        "non_vulnerable": negative,
        "positive_rate": positive / len(values) if values else 0.0,
        "imbalance_ratio": negative / positive if positive else float("inf"),
    }

