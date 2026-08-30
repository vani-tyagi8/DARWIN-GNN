from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


class Devign(nn.Module):
    """Devign GGNN plus dual convolutional graph-classification head."""

    def __init__(self, input_dim: int = 100, hidden_dim: int = 200,
                 num_steps: int = 6, num_edge_types: int = 6) -> None:
        super().__init__()
        if hidden_dim < input_dim:
            raise ValueError("hidden_dim must be >= input_dim")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.num_steps = num_steps
        self.edge_weights = nn.Parameter(torch.empty(num_edge_types, hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.edge_weights)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.h_conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3)
        self.h_pool1 = nn.MaxPool1d(3, stride=2)
        self.h_conv2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.h_pool2 = nn.MaxPool1d(2, stride=2)
        concat_dim = hidden_dim + input_dim
        self.c_conv1 = nn.Conv1d(concat_dim, concat_dim, 3)
        self.c_pool1 = nn.MaxPool1d(3, stride=2)
        self.c_conv2 = nn.Conv1d(concat_dim, concat_dim, 1)
        self.c_pool2 = nn.MaxPool1d(2, stride=2)
        self.h_score = nn.Linear(hidden_dim, 1)
        self.c_score = nn.Linear(concat_dim, 1)

    def _typed_messages(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        src, dst = edge_index
        transformed = torch.bmm(x[src].unsqueeze(1), self.edge_weights[edge_type]).squeeze(1)
        messages = torch.zeros_like(x)
        messages.index_add_(0, dst, transformed.to(dtype=x.dtype))
        return messages

    def forward(self, data) -> Tensor:
        original = data.x
        # DGL's reference GatedGraphConv zero-pads inputs when out_feats is
        # larger than in_feats; it does not learn an input projection.
        hidden = F.pad(original, (0, self.hidden_dim - self.input_dim))
        for _ in range(self.num_steps):
            messages = self._typed_messages(hidden, data.edge_index, data.edge_type)
            hidden = self.gru(messages, hidden)
        dense_h, _ = to_dense_batch(hidden, data.batch)
        dense_x, _ = to_dense_batch(original, data.batch)
        # The original two pooling stages need at least eight positions. This
        # pad only affects unusually small graphs and prevents invalid kernels.
        if dense_h.size(1) < 8:
            padding = 8 - dense_h.size(1)
            dense_h = F.pad(dense_h, (0, 0, 0, padding))
            dense_x = F.pad(dense_x, (0, 0, 0, padding))
        combined = torch.cat([dense_h, dense_x], dim=-1)
        h = self.h_pool1(F.relu(self.h_conv1(dense_h.transpose(1, 2))))
        h = self.h_pool2(F.relu(self.h_conv2(h))).transpose(1, 2)
        c = self.c_pool1(F.relu(self.c_conv1(combined.transpose(1, 2))))
        c = self.c_pool2(F.relu(self.c_conv2(c))).transpose(1, 2)
        return (self.h_score(h) * self.c_score(c)).mean(dim=1).squeeze(-1)
