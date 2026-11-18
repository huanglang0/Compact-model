import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MLPModel(nn.Module):
    def __init__(self, input_dim=15, target_dim=9, d_model=128, nhead=None,
                 num_layers=6, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.Sequential(*[
            MLPBlock(d_model, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, target_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        hidden = self.input_layer(x)
        hidden = self.blocks(hidden)
        return self.output_layer(hidden)


TransformerEncoderModel = MLPModel
