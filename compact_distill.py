"""Distill the pretrained/fine-tuned four-output teacher into a small QS model."""

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from compact_model import CompactQSModel
from config import (
    COMPACT_BASE_TO_FT_RATIO, COMPACT_BATCH_SIZE, COMPACT_BUNDLE_PATH,
    COMPACT_DERIVATIVE_LAMBDA, COMPACT_DISTILL_LAMBDA, COMPACT_EPOCHS,
    COMPACT_HIDDEN_DIMS, COMPACT_LEARNING_RATE, COMPACT_PATIENCE,
    COMPACT_VA_PATH, COMPACT_WEIGHT_DECAY, COMPACT_ZERO_VDS_LAMBDA,
    DEVICE, FINETUNE_MODEL_PATH,
)
from export_compact_qs_va import export_bundle


def _serialize_scaler(scaler):
    item = {
        'kind': 'signed_log10' if getattr(scaler, 'is_log10', False) else 'standard',
        'mean': float(scaler.mean_[0]),
        'scale': float(scaler.scale_[0]),
    }
    if item['kind'] == 'signed_log10':
        item['epsilon'] = float(scaler.epsilon)
        item['high_value_threshold'] = float(scaler.high_value_threshold)
    return item


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _validation_loss(model, x, y):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(x), COMPACT_BATCH_SIZE):
            xb = torch.as_tensor(
                x[start:start + COMPACT_BATCH_SIZE], dtype=torch.float32, device=DEVICE
            )
            yb = torch.as_tensor(
                y[start:start + COMPACT_BATCH_SIZE], dtype=torch.float32, device=DEVICE
            )
            total += float(F.mse_loss(model(xb), yb, reduction='sum'))
            count += int(yb.numel())
    return total / max(count, 1)


def train_compact_student(
    teacher,
    x_pretrain_scaled,
    y_pretrain_scaled,
    x_ft_scaled,
    y_ft_scaled,
    x_valid_scaled,
    y_valid_scaled,
    scaler_x,
    scalers_y,
    input_names,
    target_names,
    raw_domain_data,
):
    """Train, bundle and export the lightweight student.

    One output Jacobian is matched per batch, rotating through Qg/Qd/Qs/Ids.
    This preserves AC derivatives without making distillation prohibitively slow.
    """
    if list(target_names) != ['QG_TOTAL', 'QD_TOTAL', 'QS_TOTAL', 'ids']:
        raise ValueError('Compact QS deployment requires Qg, Qd, Qs and Ids targets')

    rng = np.random.default_rng(42)
    base_count = min(
        len(x_pretrain_scaled), COMPACT_BASE_TO_FT_RATIO * len(x_ft_scaled)
    )
    base_indices = rng.choice(len(x_pretrain_scaled), size=base_count, replace=False)
    x_train = np.concatenate([x_ft_scaled, x_pretrain_scaled[base_indices]], axis=0)
    y_train = np.concatenate([y_ft_scaled, y_pretrain_scaled[base_indices]], axis=0)
    dataset = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=COMPACT_BATCH_SIZE, shuffle=True)

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = CompactQSModel(
        input_dim=len(input_names), hidden_dims=COMPACT_HIDDEN_DIMS,
        output_dim=len(target_names),
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=COMPACT_LEARNING_RATE,
        weight_decay=COMPACT_WEIGHT_DECAY,
    )

    zero_vds_scaled = float((0.0 - scaler_x.mean_[1]) / scaler_x.scale_[1])
    ids_index = target_names.index('ids')
    ids_scaler = scalers_y[ids_index]
    zero_ids_scaled = float((0.0 - ids_scaler.mean_[0]) / ids_scaler.scale_[0])

    best_loss = float('inf')
    best_state = None
    stale_epochs = 0
    global_batch = 0
    progress = tqdm(range(COMPACT_EPOCHS), desc='distill compact QS model', unit='epoch')
    for _epoch in progress:
        student.train()
        train_total = 0.0
        batch_count = 0
        for x_cpu, y_cpu in loader:
            x_batch = x_cpu.to(DEVICE).requires_grad_(COMPACT_DERIVATIVE_LAMBDA > 0)
            y_batch = y_cpu.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            student_output = student(x_batch)
            teacher_output = teacher(x_batch)

            supervised = F.smooth_l1_loss(student_output, y_batch)
            distillation = F.smooth_l1_loss(student_output, teacher_output.detach())
            derivative = torch.zeros((), device=DEVICE)
            if COMPACT_DERIVATIVE_LAMBDA > 0:
                output_index = global_batch % len(target_names)
                student_jacobian = torch.autograd.grad(
                    student_output[:, output_index].sum(), x_batch,
                    create_graph=True, retain_graph=True,
                )[0][:, :3]
                teacher_jacobian = torch.autograd.grad(
                    teacher_output[:, output_index].sum(), x_batch,
                    create_graph=False, retain_graph=False,
                )[0][:, :3].detach()
                derivative = F.smooth_l1_loss(student_jacobian, teacher_jacobian)

            zero_input = x_batch.detach().clone()
            zero_input[:, 1] = zero_vds_scaled
            zero_current = student(zero_input)[:, ids_index]
            zero_loss = F.smooth_l1_loss(
                zero_current, torch.full_like(zero_current, zero_ids_scaled)
            )
            loss = (
                supervised
                + COMPACT_DISTILL_LAMBDA * distillation
                + COMPACT_DERIVATIVE_LAMBDA * derivative
                + COMPACT_ZERO_VDS_LAMBDA * zero_loss
            )
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach())
            batch_count += 1
            global_batch += 1

        valid_loss = _validation_loss(student, x_valid_scaled, y_valid_scaled)
        progress.set_postfix(
            train=f'{train_total / max(batch_count, 1):.3e}', valid=f'{valid_loss:.3e}'
        )
        if valid_loss < best_loss - 1e-8:
            best_loss = valid_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in student.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= COMPACT_PATIENCE:
                break
    progress.close()
    if best_state is None:
        raise RuntimeError('Compact student did not produce a valid checkpoint')
    student.load_state_dict(best_state)

    raw_domain_data = np.asarray(raw_domain_data, dtype=np.float64)
    bundle = {
        'format_version': 1,
        'model_class': 'CompactQSModel',
        'model': {
            'input_dim': len(input_names),
            'hidden_dims': tuple(COMPACT_HIDDEN_DIMS),
            'output_dim': len(target_names),
            'activation': 'tanh',
        },
        'model_state_dict': best_state,
        'features': list(input_names),
        'targets': list(target_names),
        'input_scaler': {
            'mean': scaler_x.mean_.astype(float).tolist(),
            'scale': scaler_x.scale_.astype(float).tolist(),
        },
        'output_scalers': [_serialize_scaler(scaler) for scaler in scalers_y],
        'training_domain': {
            'minimum': raw_domain_data.min(axis=0).astype(float).tolist(),
            'maximum': raw_domain_data.max(axis=0).astype(float).tolist(),
        },
        'teacher_checkpoint': str(FINETUNE_MODEL_PATH),
        'teacher_sha256': _sha256(FINETUNE_MODEL_PATH),
        'compact_validation_scaled_mse': float(best_loss),
    }
    destination = Path(COMPACT_BUNDLE_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, destination)
    va_path = export_bundle(destination, COMPACT_VA_PATH)
    return student, destination, va_path, best_loss
