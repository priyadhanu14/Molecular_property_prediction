from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.nn.models import AttentiveFP


def supported_models() -> tuple[str, ...]:
    return ("attentivefp", "gine")


class GINEClassifier(nn.Module):
    """Simple GINE baseline classifier for graph-level binary classification."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_embedding = global_add_pool(x, batch)
        return self.head(graph_embedding)


def build_model(
    model_name: str,
    node_dim: int,
    edge_dim: int,
    num_classes: int = 2,
    hidden_dim: int = 128,
    num_layers: int = 3,
    dropout: float = 0.2,
    num_timesteps: int = 2,
) -> nn.Module:
    model_key = model_name.lower()
    if model_key == "attentivefp":
        return AttentiveFP(
            in_channels=node_dim,
            hidden_channels=hidden_dim,
            out_channels=num_classes,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
    if model_key == "gine":
        return GINEClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model '{model_name}'. Supported: {supported_models()}")

