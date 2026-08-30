from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from .data import (IndexedJsonlGraphDataset, class_distribution, labels_for,
                   load_split, validate_graphs, validate_split_bundle)
from .losses import make_loss
from .metrics import binary_metrics
from .model import Devign
from .sampling import sampled_indices


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def predict(model, loader, device) -> tuple[list[int], list[float], list[str]]:
    model.eval()
    labels, probabilities, sample_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probability = torch.sigmoid(model(batch))
            labels.extend(batch.y.int().cpu().tolist())
            probabilities.extend(probability.cpu().tolist())
            ids = batch.sample_id if isinstance(batch.sample_id, list) else [batch.sample_id]
            sample_ids.extend(map(str, ids))
    return labels, probabilities, sample_ids


def run(config: dict) -> dict:
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(config.get("device", "auto"))
    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["training"]
    root = Path(data_cfg["root"]) / data_cfg["dataset"]
    train_graphs = load_split(root, "train")
    valid_graphs = load_split(root, "valid")
    test_graphs = load_split(root, "test")
    for graphs in (train_graphs, valid_graphs, test_graphs):
        if isinstance(graphs, IndexedJsonlGraphDataset):
            if graphs.feature_dim != int(model_cfg["input_dim"]):
                raise ValueError("dataset feature width does not match model configuration")
            if graphs.num_edge_types != int(model_cfg["num_edge_types"]):
                raise ValueError("dataset edge vocabulary does not match model configuration")
            validate_graphs([graphs[0], graphs[len(graphs) - 1]], int(model_cfg["input_dim"]),
                            int(model_cfg["num_edge_types"]))
        else:
            validate_graphs(graphs, int(model_cfg["input_dim"]), int(model_cfg["num_edge_types"]))
    validate_split_bundle(train_graphs, valid_graphs, test_graphs)
    labels = torch.tensor(labels_for(train_graphs))
    indices = sampled_indices(labels.numpy(), train_cfg["strategy"], seed)
    sampled_train = Subset(train_graphs, indices)
    num_workers = int(data_cfg.get("num_workers", 0))
    loader_args = dict(
        batch_size=int(data_cfg["batch_size"]),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(sampled_train, shuffle=True, **loader_args)
    valid_loader = DataLoader(valid_graphs, shuffle=False, **loader_args)
    test_loader = DataLoader(test_graphs, shuffle=False, **loader_args)
    model = Devign(**model_cfg).to(device)
    loss_fn = make_loss(train_cfg["strategy"], labels, train_cfg.get("focal_gamma", 2.0),
                        train_cfg.get("focal_alpha")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]),
                                 weight_decay=float(train_cfg["weight_decay"]))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_state, best_score, stale = None, -1.0, 0
    max_epochs = int(train_cfg["epochs"])
    for epoch in range(max_epochs):
        model.train()
        epoch_loss, batch_count = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss = loss_fn(model(batch), batch.y.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
            batch_count += 1
        val_y, val_prob, _ = predict(model, valid_loader, device)
        score = float(binary_metrics(val_y, val_prob, train_cfg["threshold"])["pr_auc"])
        if score > best_score:
            best_score, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(
            f"Epoch {epoch + 1}/{max_epochs} | loss={epoch_loss / max(batch_count, 1):.6f} "
            f"| val_pr_auc={score:.6f} | best={best_score:.6f} | patience={stale}",
            flush=True,
        )
        if stale >= int(train_cfg["patience"]):
            print(f"Early stopping after epoch {epoch + 1}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    y_true, y_prob, sample_ids = predict(model, test_loader, device)
    metrics = binary_metrics(y_true, y_prob, train_cfg["threshold"])
    result = {
        "dataset": data_cfg["dataset"], "strategy": train_cfg["strategy"], "seed": seed,
        "device": str(device), **metrics, "train_distribution": class_distribution(train_graphs),
        "sampled_train_distribution": class_distribution(labels=labels[indices].int().tolist()),
        "test_distribution": class_distribution(test_graphs),
    }
    run_dir = Path(config["output_dir"]) / data_cfg["dataset"] / train_cfg["strategy"] / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "best_model.pt")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "label", "probability"])
        writer.writerows(zip(sample_ids, y_true, y_prob))
    return result
