"""Small, smooth network intended for Verilog-A deployment."""

import torch
import torch.nn as nn


class CompactQSModel(nn.Module):
    """Two-layer tanh MLP for [Qg, Qd, Qs, Ids].

    Tanh is intentionally used because it is smooth for simulator Jacobians and
    substantially cheaper than the teacher model's GELU + LayerNorm blocks.
    """

    def __init__(self, input_dim=6, hidden_dims=(24, 24), output_dim=4):
        super().__init__()
        h1, h2 = hidden_dims
        self.input_dim = int(input_dim)
        self.hidden_dims = (int(h1), int(h2))
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, self.output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)
