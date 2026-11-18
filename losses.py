import torch
import torch.nn as nn
from config import HIGH_VALUE_LOSS_LAMBDA


class NAE_NSELoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8, nae_weight: float = 0.7, nse_weight: float = 0.3):
        super().__init__()
        self.epsilon = epsilon
        self.nae_weight = nae_weight
        self.nse_weight = nse_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if y_true.shape[0] <= 1:
            return torch.mean(torch.abs(y_pred - y_true))

        mean_true = y_true.mean(dim=0, keepdim=True)
        std_true = y_true.std(dim=0, keepdim=True) + self.epsilon

        y_true_norm = (y_true - mean_true) / std_true
        y_pred_norm = (y_pred - mean_true) / std_true

        err = y_pred_norm - y_true_norm
        nae = torch.mean(torch.abs(err))
        nse = torch.mean(err ** 2)

        return self.nae_weight * nae + self.nse_weight * nse


class WeightedBNILoss(nn.Module):
    def __init__(self, target_names, scalers_y=None, epsilon: float = 1e-12,
                 high_current_lambda: float = HIGH_VALUE_LOSS_LAMBDA,
                 max_log_abs: float = 30.0):
        super().__init__()
        self.base_loss = NAE_NSELoss()
        self.epsilon = epsilon
        self.high_current_lambda = high_current_lambda
        self.max_log_abs = max_log_abs

        indices = []
        means = []
        scales = []
        log_epsilons = []
        thresholds = []

        if scalers_y is not None:
            for i, scaler in enumerate(scalers_y):
                if getattr(scaler, "is_log10", False):
                    indices.append(i)
                    means.append(float(scaler.mean_[0]))
                    scales.append(float(scaler.scale_[0]))
                    log_epsilons.append(float(scaler.epsilon))
                    thresholds.append(float(scaler.high_value_threshold))

        self.target_names = target_names
        self.register_buffer("log_indices", torch.tensor(indices, dtype=torch.long))
        self.register_buffer("log_means", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("log_scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("log_epsilons", torch.tensor(log_epsilons, dtype=torch.float32))
        self.register_buffer("high_value_thresholds", torch.tensor(thresholds, dtype=torch.float32))

    def _restore_signed_log10_value(self, y_scaled: torch.Tensor) -> torch.Tensor:
        log_value = y_scaled * self.log_scales + self.log_means
        log_abs = torch.clamp(torch.abs(log_value), max=self.max_log_abs)
        return torch.sign(log_value) * self.log_epsilons * (torch.pow(10.0, log_abs) - 1.0)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor = None):
        base_loss = self.base_loss(y_pred, y_true)

        if self.log_indices.numel() == 0 or self.high_current_lambda <= 0:
            return base_loss

        pred_log_scaled = y_pred.index_select(1, self.log_indices)
        true_log_scaled = y_true.index_select(1, self.log_indices)

        pred_value = self._restore_signed_log10_value(pred_log_scaled)
        true_value = self._restore_signed_log10_value(true_log_scaled)

        high_mask = torch.abs(true_value) >= self.high_value_thresholds
        if torch.any(high_mask):
            rel_err = torch.abs(pred_value - true_value) / (torch.abs(true_value) + self.epsilon)
            high_value_loss = torch.mean(rel_err[high_mask])
        else:
            high_value_loss = torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)

        return base_loss + self.high_current_lambda * high_value_loss
