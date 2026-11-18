"""Export the full fine-tuned MLP directly as a charge-based Verilog-A model."""

from pathlib import Path

import numpy as np


LN10 = 2.302585092994046
GELU_FACTOR = 0.7978845608028654
GELU_CUBIC = 0.044715


def _real(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite Verilog-A constant: {value}")
    return f"{value:.17e}"


def _array(layer, name):
    return getattr(layer, name).detach().cpu().double().numpy()


def _linear_assignment(lines, destination, bias, weights, source, activation=None,
                       residual=None):
    opening = f"{activation}(" if activation else ""
    first_term = _real(bias)
    if residual is not None:
        first_term = f"{residual} + {first_term}"
    lines.append(f"    {destination} = {opening}{first_term}")
    for index, coefficient in enumerate(weights):
        lines.append(f"        + ({_real(coefficient)}) * {source}[{index}]")
    lines.append("        );" if activation else "        ;")


def _layer_norm(lines, layer, source, destination):
    width = int(layer.normalized_shape[0])
    gamma = _array(layer, "weight")
    beta = _array(layer, "bias")

    lines.append("    ln_mean = 0.0;")
    for index in range(width):
        lines.append(f"    ln_mean = ln_mean + {source}[{index}];")
    lines.append(f"    ln_mean = ln_mean / {_real(width)};")
    lines.append("    ln_variance = 0.0;")
    for index in range(width):
        lines.append(
            f"    ln_variance = ln_variance + ({source}[{index}] - ln_mean)"
            f" * ({source}[{index}] - ln_mean);"
        )
    lines.append(f"    ln_variance = ln_variance / {_real(width)};")
    lines.append(f"    ln_inverse_std = 1.0 / sqrt(ln_variance + {_real(layer.eps)});")
    for index in range(width):
        lines.append(
            f"    {destination}[{index}] = ({_real(gamma[index])})"
            f" * ({source}[{index}] - ln_mean) * ln_inverse_std"
            f" + ({_real(beta[index])});"
        )


def export_finetuned_model(model, scaler_x, scalers_y, input_names, target_names,
                           raw_domain_data, output_path, module_name):
    expected_inputs = ["vgs", "vds", "vbs", "length", "nfin", "temp"]
    expected_targets = ["QG_TOTAL", "QD_TOTAL", "QS_TOTAL", "ids"]
    if list(input_names) != expected_inputs:
        raise ValueError(f"Unsupported input order: {input_names}")
    if list(target_names) != expected_targets:
        raise ValueError(f"Unsupported target order: {target_names}")

    model.eval()
    input_linear = model.input_layer[0]
    output_norm = model.output_layer[0]
    output_linear = model.output_layer[1]
    d_model = int(input_linear.out_features)
    if any(int(block.net[0].normalized_shape[0]) != d_model for block in model.blocks):
        raise ValueError("All residual blocks must use the same hidden width")
    max_feedforward = max(int(block.net[1].out_features) for block in model.blocks)

    x_mean = np.asarray(scaler_x.mean_, dtype=np.float64)
    x_scale = np.asarray(scaler_x.scale_, dtype=np.float64)
    if np.any(x_scale <= 0):
        raise ValueError("Input scaler contains a non-positive scale")

    input_weight = _array(input_linear, "weight")
    input_bias = _array(input_linear, "bias")
    input_weight = input_weight / x_scale[np.newaxis, :]
    input_bias = input_bias - np.sum(
        _array(input_linear, "weight") * (x_mean / x_scale)[np.newaxis, :], axis=1
    )

    output_mean = np.asarray([float(s.mean_[0]) for s in scalers_y], dtype=np.float64)
    output_scale = np.asarray([float(s.scale_[0]) for s in scalers_y], dtype=np.float64)
    output_weight = _array(output_linear, "weight") * output_scale[:, np.newaxis]
    output_bias = _array(output_linear, "bias") * output_scale + output_mean
    ids_scaler = scalers_y[3]
    if not getattr(ids_scaler, "is_log10", False):
        raise ValueError("Ids output must use signed-log10 scaling")
    ids_epsilon = float(ids_scaler.epsilon)

    raw_domain_data = np.asarray(raw_domain_data, dtype=np.float64)
    domain_min = raw_domain_data.min(axis=0).astype(float).tolist()
    domain_max = raw_domain_data.max(axis=0).astype(float).tolist()

    lines = [
        '`include "constants.vams"',
        '`include "disciplines.vams"',
        "",
        "// Full fine-tuned quasi-static neural compact model; no distillation.",
        "// PyTorch GELU is represented with its standard smooth tanh approximation.",
        "// DC: Ids contribution. AC/transient: terminal-charge ddt() contributions.",
        "// Validated operation should remain inside the training domain below.",
        f"// Training minimum: {domain_min}",
        f"// Training maximum: {domain_max}",
        f"module {module_name}(d, g, s, b);",
        "  inout d, g, s, b;",
        "  electrical d, g, s, b;",
        f"  parameter real length = {_real(x_mean[3])};",
        f"  parameter real nfin = {_real(x_mean[4])};",
        f"  parameter real temp = {_real(x_mean[5])};",
        f"  real input_value[0:{len(expected_inputs) - 1}];",
        f"  real hidden[0:{d_model - 1}];",
        f"  real normalized[0:{d_model - 1}];",
        f"  real feedforward[0:{max_feedforward - 1}];",
        f"  real projected[0:{d_model - 1}];",
        "  real ln_mean, ln_variance, ln_inverse_std;",
        "  real qg, qd, qs, qb;",
        "  real ids_log, ids_abs_log, ids_value;",
        "",
        "  analog function real gelu_tanh;",
        "    input value;",
        "    real value;",
        "    begin",
        f"      gelu_tanh = 0.5 * value * (1.0 + tanh({_real(GELU_FACTOR)}",
        f"                  * (value + {_real(GELU_CUBIC)} * value * value * value)));",
        "    end",
        "  endfunction",
        "",
        "  analog begin",
        "    input_value[0] = V(g, s);",
        "    input_value[1] = V(d, s);",
        "    input_value[2] = V(b, s);",
        "    input_value[3] = length;",
        "    input_value[4] = nfin;",
        "    input_value[5] = temp;",
    ]

    for row in range(d_model):
        _linear_assignment(
            lines, f"hidden[{row}]", input_bias[row], input_weight[row],
            "input_value", activation="gelu_tanh"
        )

    for block in model.blocks:
        layer_norm = block.net[0]
        first = block.net[1]
        second = block.net[4]
        _layer_norm(lines, layer_norm, "hidden", "normalized")
        first_weight = _array(first, "weight")
        first_bias = _array(first, "bias")
        for row in range(first.out_features):
            _linear_assignment(
                lines, f"feedforward[{row}]", first_bias[row], first_weight[row],
                "normalized", activation="gelu_tanh"
            )
        second_weight = _array(second, "weight")
        second_bias = _array(second, "bias")
        for row in range(d_model):
            _linear_assignment(
                lines, f"projected[{row}]", second_bias[row], second_weight[row],
                "feedforward", residual=f"hidden[{row}]"
            )
        for row in range(d_model):
            lines.append(f"    hidden[{row}] = projected[{row}];")

    _layer_norm(lines, output_norm, "hidden", "normalized")
    output_names = ["qg", "qd", "qs", "ids_log"]
    for row, output_name in enumerate(output_names):
        _linear_assignment(
            lines, output_name, output_bias[row], output_weight[row], "normalized"
        )

    lines.extend([
        "    qb = -(qg + qd + qs);",
        "    ids_abs_log = sqrt(ids_log * ids_log + 1.0e-24);",
        f"    ids_value = ({_real(ids_epsilon)}) * (ids_log / ids_abs_log)",
        f"                * (limexp({_real(LN10)} * ids_abs_log) - 1.0);",
        "",
        "    I(d, s) <+ ids_value;",
        "    I(g, s) <+ ddt(qg);",
        "    I(d, s) <+ ddt(qd);",
        "    I(b, s) <+ ddt(qb);",
        "  end",
        "endmodule",
        "",
    ])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
