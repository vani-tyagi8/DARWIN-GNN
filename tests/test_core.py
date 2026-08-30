import numpy as np
import pytest
import torch
import json
import os
from torch_geometric.data import Batch, Data

from devign_imbalance.losses import FocalLoss, make_loss
from devign_imbalance.metrics import binary_metrics
from devign_imbalance.sampling import sampled_indices
from devign_imbalance.model import Devign
from devign_imbalance.data import IndexedJsonlGraphDataset, validate_split_bundle


def test_sampling_balances_both_strategies():
    labels = np.array([0] * 8 + [1] * 2)
    for strategy, expected in [("random_oversampling", 16), ("random_undersampling", 4)]:
        chosen = sampled_indices(labels, strategy, seed=7)
        sampled = labels[chosen]
        assert len(chosen) == expected
        assert np.sum(sampled == 0) == np.sum(sampled == 1)


def test_sampling_is_deterministic():
    labels = [0, 0, 0, 1]
    assert sampled_indices(labels, "random_oversampling", 3) == sampled_indices(labels, "random_oversampling", 3)


def test_metrics_known_example():
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9])
    assert result["tp"] == 2 and result["fp"] == 1
    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(2 / 3)


def test_focal_loss_is_finite_and_differentiable():
    logits = torch.tensor([0.2, -1.0], requires_grad=True)
    loss = FocalLoss()(logits, torch.tensor([1.0, 0.0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_class_weight_ratio():
    loss = make_loss("class_weighted", torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert loss.pos_weight.item() == 3.0


def test_devign_forward_returns_one_logit_per_graph():
    graph = Data(
        x=torch.randn(5, 4),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        edge_type=torch.tensor([0, 1, 0, 1]),
        y=torch.tensor(1.0),
    )
    batch = Batch.from_data_list([graph, graph.clone()])
    logits = Devign(input_dim=4, hidden_dim=8, num_steps=2, num_edge_types=2)(batch)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_devign_forward_supports_mixed_precision_autocast():
    graph = Data(
        x=torch.randn(8, 4),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        edge_type=torch.tensor([0, 1, 0, 1]),
        y=torch.tensor(1.0),
    )
    batch = Batch.from_data_list([graph])
    model = Devign(input_dim=4, hidden_dim=8, num_steps=2, num_edge_types=2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(batch)
    assert logits.shape == (1,)
    assert torch.isfinite(logits).all()


def test_split_validation_rejects_leakage():
    def item(sample_id, label):
        graph = Data(y=torch.tensor(float(label)))
        graph.sample_id = sample_id
        return graph
    train = [item("shared", 0), item("train-positive", 1)]
    valid = [item("shared", 0), item("valid-positive", 1)]
    test = [item("test-negative", 0), item("test-positive", 1)]
    with pytest.raises(ValueError, match="more than one split"):
        validate_split_bundle(train, valid, test)


def test_indexed_jsonl_dataset_reads_by_offset(tmp_path):
    records = [
        {"id": "a", "node_features": [[0.0, 1.0]], "graph": [[0, 0, 0]], "target": 0},
        {"id": "b", "node_features": [[1.0, 0.0]], "graph": [], "target": 1},
    ]
    data_path = tmp_path / "train.jsonl"
    offsets = []
    with data_path.open("wb") as stream:
        for record in records:
            offsets.append(stream.tell())
            stream.write(json.dumps(record).encode() + b"\n")
    index_path = tmp_path / "train.index.json"
    index_path.write_text(json.dumps({
        "offsets": offsets, "labels": [0, 1], "id_prefix": "train",
        "feature_dim": 2, "num_edge_types": 1,
    }))
    dataset = IndexedJsonlGraphDataset(data_path, index_path)
    assert len(dataset) == 2
    assert dataset[1].sample_id == "b"
    assert dataset[1].x.tolist() == [[1.0, 0.0]]


def test_indexed_jsonl_dataset_reopens_in_worker_process(tmp_path, monkeypatch):
    record = {"id": "a", "node_features": [[0.0]], "graph": [], "target": 0}
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    index_path = tmp_path / "train.index.json"
    index_path.write_text(json.dumps({
        "offsets": [0], "labels": [0], "id_prefix": "train",
        "feature_dim": 1, "num_edge_types": 1,
    }))
    dataset = IndexedJsonlGraphDataset(data_path, index_path)
    assert dataset[0].sample_id == "a"
    inherited_stream = dataset._stream
    original_pid = os.getpid()
    monkeypatch.setattr("devign_imbalance.data.os.getpid", lambda: original_pid + 1)
    assert dataset[0].sample_id == "a"
    assert inherited_stream.closed

