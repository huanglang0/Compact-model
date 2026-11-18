"""Export a CompactQSModel bundle as a lightweight charge-based Verilog-A model."""

import argparse
from pathlib import Path

import numpy as np
import torch

from compact_model import CompactQSModel
from config import COMPACT_BUNDLE_PATH, COMPACT_VA_PATH


LN10 = 2.302585092994046


def _real(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f'Non-finite Verilog-A constant: {value}')
    return f'{value:.17e}'


def _assignment(lines, destination, bias, weights, source, activation=None):
    opening = f'{activation}(' if activation else ''
    lines.append(f'    {destination} = {opening}{_real(bias)}')
    for index, coefficient in enumerate(weights):
        lines.append(f'        + ({_real(coefficient)}) * {source}[{index}]')
    lines.append('        );' if activation else '        ;')


def export_bundle(bundle_path, output_path):
    bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
    if bundle.get('model_class') != 'CompactQSModel':
        raise ValueError('Bundle is not a CompactQSModel deployment bundle')
    if bundle.get('features') != ['vgs', 'vds', 'vbs', 'length', 'nfin', 'temp']:
        raise ValueError(f"Unsupported features: {bundle.get('features')}")
    if bundle.get('targets') != ['QG_TOTAL', 'QD_TOTAL', 'QS_TOTAL', 'ids']:
        raise ValueError(f"Unsupported targets: {bundle.get('targets')}")

    metadata = bundle['model']
    model = CompactQSModel(
        input_dim=metadata['input_dim'],
        hidden_dims=tuple(metadata['hidden_dims']),
        output_dim=metadata['output_dim'],
    )
    model.load_state_dict(bundle['model_state_dict'])
    model.eval()

    first = model.network[0]
    second = model.network[2]
    output = model.network[4]
    w0 = first.weight.detach().double().numpy()
    b0 = first.bias.detach().double().numpy()
    w1 = second.weight.detach().double().numpy()
    b1 = second.bias.detach().double().numpy()
    w2 = output.weight.detach().double().numpy()
    b2 = output.bias.detach().double().numpy()

    x_mean = np.asarray(bundle['input_scaler']['mean'], dtype=np.float64)
    x_scale = np.asarray(bundle['input_scaler']['scale'], dtype=np.float64)
    if np.any(x_scale <= 0):
        raise ValueError('Input scaler contains a non-positive scale')

    # Fold StandardScaler into the first layer: W((x-mu)/sigma)+b.
    w0_folded = w0 / x_scale[np.newaxis, :]
    b0_folded = b0 - np.sum(w0 * (x_mean / x_scale)[np.newaxis, :], axis=1)

    # Fold each output StandardScaler into the last linear layer. For Ids this
    # produces signed-log10 current, which is inverted below with limexp().
    output_scalers = bundle['output_scalers']
    output_mean = np.asarray([item['mean'] for item in output_scalers])
    output_scale = np.asarray([item['scale'] for item in output_scalers])
    w2_folded = w2 * output_scale[:, np.newaxis]
    b2_folded = b2 * output_scale + output_mean
    ids_scaler = output_scalers[3]
    if ids_scaler.get('kind') != 'signed_log10':
        raise ValueError('Ids output must use signed_log10 scaling')
    ids_epsilon = float(ids_scaler['epsilon'])

    h1, h2 = metadata['hidden_dims']
    domain = bundle.get('training_domain', {})
    domain_min = domain.get('minimum', [])
    domain_max = domain.get('maximum', [])
    lines = [
        '`include "constants.vams"',
        '`include "disciplines.vams"',
        '',
        '// Lightweight quasi-static neural compact model.',
        '// DC: Ids contribution. AC/transient: terminal-charge ddt() contributions.',
        '// Validated operation should remain inside the training-domain comments below.',
        f'// Training minimum: {domain_min}',
        f'// Training maximum: {domain_max}',
        'module compact_qs_finfet(d, g, s, b);',
        '  inout d, g, s, b;',
        '  electrical d, g, s, b;',
        f'  parameter real length = {_real(x_mean[3])};',
        f'  parameter real nfin = {_real(x_mean[4])};',
        f'  parameter real temp = {_real(x_mean[5])};',
        f'  real input_value[0:{metadata["input_dim"] - 1}];',
        f'  real hidden1[0:{h1 - 1}];',
        f'  real hidden2[0:{h2 - 1}];',
        '  real qg, qd, qs, qb;',
        '  real ids_log, ids_abs_log, ids_value;',
        '',
        '  analog begin',
        '    input_value[0] = V(g, s);',
        '    input_value[1] = V(d, s);',
        '    input_value[2] = V(b, s);',
        '    input_value[3] = length;',
        '    input_value[4] = nfin;',
        '    input_value[5] = temp;',
    ]
    for row in range(h1):
        _assignment(
            lines, f'hidden1[{row}]', b0_folded[row], w0_folded[row],
            'input_value', activation='tanh',
        )
    for row in range(h2):
        _assignment(
            lines, f'hidden2[{row}]', b1[row], w1[row],
            'hidden1', activation='tanh',
        )
    output_names = ['qg', 'qd', 'qs', 'ids_log']
    for row, name in enumerate(output_names):
        _assignment(lines, name, b2_folded[row], w2_folded[row], 'hidden2')

    # Smooth absolute value makes the signed-log inverse differentiable around
    # zero. limexp improves Newton convergence and prevents numerical overflow.
    lines.extend([
        '    qb = -(qg + qd + qs);',
        '    ids_abs_log = sqrt(ids_log * ids_log + 1.0e-24);',
        f'    ids_value = ({_real(ids_epsilon)}) * (ids_log / ids_abs_log)',
        f'                * (limexp({_real(LN10)} * ids_abs_log) - 1.0);',
        '',
        '    // Static channel current: active in DC, AC operating point and transient.',
        '    I(d, s) <+ ids_value;',
        '    // Charge formulation: zero in DC, automatically linearized in AC,',
        '    // and differentiated by the simulator during transient analysis.',
        '    I(g, s) <+ ddt(qg);',
        '    I(d, s) <+ ddt(qd);',
        '    I(b, s) <+ ddt(qb);',
        '  end',
        'endmodule',
        '',
    ])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', default=COMPACT_BUNDLE_PATH)
    parser.add_argument('--output', default=COMPACT_VA_PATH)
    arguments = parser.parse_args()
    destination = export_bundle(arguments.bundle, arguments.output)
    print(f'Verilog-A model: {destination}')


if __name__ == '__main__':
    main()
