"""Read two CSVs and compare numerical dIds/dVgs without training dependencies.

Edit the paths below, then run: python compare_csv_derivatives.py
"""
import csv
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ===== 在这里填写文件路径、列名 =====
TRUE_CSV_PATH = r"/home/huangl/myproject/mos_model_nn/data/Test/finfet_7nm_20000/output.csv"
PRED_CSV_PATH = r"/home/huangl/myproject/mos_model_nn/DAC/script_MLP_Log_FT/picture/predictions.csv"
OUTPUT_DIR = r"/home/huangl/myproject/mos_model_nn/DAC/script_MLP_Log_FT/derivative_comparison"

TRUE_VGS_COLUMN = 'vgs'
TRUE_IDS_COLUMN = 'ids'
PRED_VGS_COLUMN = 'vgs'
PRED_IDS_COLUMN = 'ids'  # 如果预测列叫 ids_pred，在这里修改
CSV_ENCODING = 'utf-8-sig'
GM_FLOOR = None  # 相对误差分母下限，单位 A/V；None 表示自动设置

# 除 Vgs 外，用这些输入条件区分不同扫描曲线。
GROUP_COLUMNS = ['vds', 'vbs', 'length', 'nfin', 'temp']
# 只有确认预测 CSV 与真实 CSV 逐行对应、未排序或删行，才能设 True。
# 用于预测 CSV 没有保存完整输入条件列的情况。
PRED_ROWS_MATCH_TRUE = False

# 计算标准 MRE 时剔除 |gm_true| 过小的点，避免除零/爆炸
MRE_MIN_ABS_GM = 1e-15


def read_table(path, voltage_column, current_column, encoding):
    with open(path, newline='', encoding=encoding) as f:
        reader = csv.DictReader(f)
        missing = {voltage_column, current_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}; found {reader.fieldnames}')
        headers = reader.fieldnames
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: empty CSV')
    for row in rows:
        for col in {voltage_column, current_column, *GROUP_COLUMNS} & set(headers):
            row[col] = float(row[col])
            if not np.isfinite(row[col]):
                raise ValueError(f'{path}: non-finite value in {col}')
    return headers, rows


def group_curves(rows, voltage_column, current_column, columns):
    groups = {}
    for row in rows:
        key = tuple(row[c] for c in columns)
        groups.setdefault(key, []).append((row[voltage_column], row[current_column]))
    result = {}
    duplicates = 0
    for key, pairs in groups.items():
        a = np.asarray(pairs, dtype=float)
        v, inverse, counts = np.unique(a[:, 0], return_inverse=True, return_counts=True)
        means = np.bincount(inverse, weights=a[:, 1]) / counts
        # Collapse only agreeing repetitions of the same complete input point.
        # Conflicting labels can indicate an omitted condition and must not be mixed.
        if not np.allclose(a[:, 1], means[inverse], rtol=1e-6, atol=1e-30):
            raise ValueError(f'Conflicting currents for identical inputs, group={key}. '
                             'Add omitted conditions to GROUP_COLUMNS or select one sweep.')
        duplicates += int(np.sum(counts-1))
        result[key] = v, means
    if duplicates:
        warnings.warn(f'Collapsed {duplicates} agreeing repeated input rows')
    return result


def compare(true_path, pred_path, output, true_vgs='vgs', pred_vgs='vgs',
            true_ids='ids', pred_ids='ids', encoding='utf-8-sig', gm_floor=None):
    ht, rt = read_table(true_path, true_vgs, true_ids, encoding)
    hp, rp = read_table(pred_path, pred_vgs, pred_ids, encoding)
    columns = [c for c in GROUP_COLUMNS if c in ht or c in hp]
    if PRED_ROWS_MATCH_TRUE:
        if len(rt) != len(rp) or any(t[true_vgs] != p[pred_vgs] for t,p in zip(rt,rp)):
            raise ValueError('Row counts or row-wise Vgs do not match; cannot copy conditions')
        for c in columns:
            if c in ht and c not in hp:
                for t,p in zip(rt,rp):
                    p[c] = t[c]
        hp = list(set(hp) | (set(columns) & set(ht)))
    for c in columns:
        # A globally constant condition can safely be attached to the other CSV.
        if c not in ht or c not in hp:
            existing, missing = (rt, rp) if c in ht else (rp, rt)
            values = {r[c] for r in existing}
            if len(values) != 1:
                raise ValueError(f'Column {c} varies but is missing from one CSV. '
                                 'Export input conditions with predictions; or set '
                                 'PRED_ROWS_MATCH_TRUE=True ONLY if rows match exactly.')
            value = next(iter(values))
            for r in missing:
                r[c] = value
    true_groups = group_curves(rt, true_vgs, true_ids, columns)
    pred_groups = group_curves(rp, pred_vgs, pred_ids, columns)
    # Only use the TRUE file's gm labels as an optional independent reference.
    reference_groups = group_curves(rt, true_vgs, 'didsdvg', columns) if 'didsdvg' in ht else None
    if true_groups.keys() != pred_groups.keys():
        raise ValueError('The CSVs have different sets of fixed-condition sweeps')
    results, manifest = [], []
    skipped = 0
    for index, (key, (vt,it)) in enumerate(sorted(true_groups.items())):
        vp,ip = pred_groups[key]
        if len(vt) < 3 or len(vp) < 3:
            skipped += 1
            continue
        folder = Path(output) / f'sweep_{index:04d}'
        reference = reference_groups[key][1] if reference_groups is not None else None
        results.append(_compare_curve(vt, it, vp, ip, folder, gm_floor, reference))
        manifest.append(dict(folder=folder.name, **dict(zip(columns,key)), points=len(vt)))
    if not results:
        raise ValueError('No fixed-condition group has >=3 unique Vgs points; '
                         'generate Vgs sweeps with other inputs fixed.')
    with (Path(output)/'sweeps.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f'Completed {len(results)} sweeps; skipped {skipped} groups with fewer than 3 Vgs points.')
    return results


def _compare_curve(vt, it, vp, ip, output, gm_floor, reference=None):
    # Align by Vgs, never by original CSV row number. Reject unequal grids
    # rather than silently introduce interpolation into the derivative error.
    if vt.shape != vp.shape or not np.array_equal(vt, vp):
        raise ValueError('The two CSVs must have the same Vgs points (row order may differ)')
    gt = np.gradient(it, vt, edge_order=2)
    gp = np.gradient(ip, vp, edge_order=2)
    error = gp - gt
    if gm_floor is None:
        gm_floor = max(float(np.max(np.abs(gt))) * 1e-3, 1e-30)
    if not np.isfinite(gm_floor) or gm_floor <= 0:
        raise ValueError('gm_floor must be positive and finite, in A/V')
    relative = np.abs(error) / np.maximum(np.abs(gt), gm_floor) * 100
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    # ===== 新增：标准 MRE（带符号），剔除 |gm_true| 过小的点 =====
    mre_mask = np.abs(gt) > MRE_MIN_ABS_GM
    if np.any(mre_mask):
        mre = float(np.mean(error[mre_mask] / gt[mre_mask]) * 100)
        mre_n_points = int(np.sum(mre_mask))
    else:
        mre = float('nan')
        mre_n_points = 0

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'derivative_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['vgs', 'ids_true', 'ids_pred', 'gm_true', 'gm_pred',
                         'gm_error_pred_minus_true', 'gm_absolute_error', 'floor_relative_error_percent'])
        writer.writerows(zip(vt, it, ip, gt, gp, error, np.abs(error), relative))
    if reference is not None:
        with (out / 'reference_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['vgs', 'true_didsdvg_reference', 'gm_true_numerical',
                             'gm_pred_numerical', 'true_numerical_minus_reference',
                             'pred_numerical_minus_reference'])
            writer.writerows(zip(vt, reference, gt, gp, gt-reference, gp-reference))
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(vt, gt, label='True dIds/dVgs')
    axes[0].plot(vt, gp, '--', label='Predicted dIds/dVgs')
    if reference is not None:
        axes[0].plot(vt, reference, ':', color='black', label='True CSV didsdvg reference')
    axes[0].set_ylabel('gm (A/V)')
    axes[0].legend()
    axes[1].plot(vt, error)
    if reference is not None:
        axes[1].plot(vt, gp-reference, color='orange', label='Pred numerical - true reference')
        axes[1].legend()
    axes[1].axhline(0, color='gray', linewidth=.8)
    axes[1].set_ylabel('Pred - true (A/V)')
    axes[2].plot(vt, relative)
    axes[2].set_ylabel('Floor relative error (%)')
    axes[2].set_xlabel('Vgs (V)')
    for ax in axes:
        ax.grid(alpha=.3)
    fig.suptitle(f'gm MAE={mae:.5g} A/V; RMSE={rmse:.5g} A/V\n'
                 f'Relative-error denominator floor={gm_floor:.5g} A/V')
    fig.tight_layout()
    fig.savefig(out / 'derivative_comparison.png', dpi=180)
    plt.close(fig)
    report = (f'Points: {len(vt)}\ngm MAE: {mae:.8g} A/V\ngm RMSE: {rmse:.8g} A/V\n'
              f'gm maximum absolute error: {np.max(np.abs(error)):.8g} A/V\n'
              f'Relative-error denominator floor: {gm_floor:.8g} A/V\n'
              f'Floor mean relative error: {np.mean(relative):.8g}%\n'
              f'MRE (signed mean relative error, |gm_true|>{MRE_MIN_ABS_GM:g}): '
              f'{mre:.8g}%  [over {mre_n_points}/{len(vt)} points]\n'
              'Endpoint derivatives use one-sided differences; interior points use central differences.\n')
    if reference is not None:
        report += (f'True numerical gm vs true didsdvg RMSE: {np.sqrt(np.mean((gt-reference)**2)):.8g} A/V\n'
                   f'Pred numerical gm vs true didsdvg RMSE: {np.sqrt(np.mean((gp-reference)**2)):.8g} A/V\n'
                   'Reference is read ONLY from true CSV; prediction CSV didsdvg is not used.\n')
    (out / 'metrics.txt').write_text(report, encoding='utf-8')
    print(report, end='')
    print(f'Saved to: {out.resolve()}')
    return gt, gp


if __name__ == '__main__':
    compare(TRUE_CSV_PATH, PRED_CSV_PATH, OUTPUT_DIR,
            TRUE_VGS_COLUMN, PRED_VGS_COLUMN,
            TRUE_IDS_COLUMN, PRED_IDS_COLUMN, CSV_ENCODING, GM_FLOOR)