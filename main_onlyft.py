# 整合所有组件的主流程：MLP一次性输出所有目标，没有自回归，没有评估阈值过滤
import os
import torch
import numpy as np
import pandas as pd
from config import *
from model import MLPModel
from data_processing import load_dataset, preprocess_data, scale_data
from train import pretrain_model, finetune_model, test_model
from metrics import calculate_metrics, calculate_average_metrics
from visualization import plot_scatter_comparison, plot_vgs_comparison, plot_loss_curves
from export_finetuned_qs_va import export_finetuned_model
from utils import set_seed, create_directories
import datetime


def save_experiment_report(global_metrics, avg_metrics, save_path):
    """保存实验报告"""
    report_content = f"""
{'='*80}
实验报告 - {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*80}

一、训练参数
{'='*40}
批次大小 (BATCH_SIZE): {BATCH_SIZE}
预训练轮数 (PRETRAIN_EPOCHS): {PRETRAIN_EPOCHS}
微调轮数 (FINETUNE_EPOCHS): {FINETUNE_EPOCHS}
预训练学习率 (LEARNING_RATE_PRETRAIN): {LEARNING_RATE_PRETRAIN}
微调学习率 (LEARNING_RATE_FINETUNE): {LEARNING_RATE_FINETUNE}
权重衰减 (WEIGHT_DECAY): {WEIGHT_DECAY}

二、Early Stopping 参数
{'='*40}
预训练耐心值 (PRETRAIN_PATIENCE): {PRETRAIN_PATIENCE}
微调耐心值 (FINETUNE_PATIENCE): {FINETUNE_PATIENCE}
改进阈值 (DELTA): {DELTA}

三、模型参数
{'='*40}
模型维度 (D_MODEL): {D_MODEL}
MLP层数 (NUM_LAYERS): {NUM_LAYERS}
前馈网络维度 (DIM_FEEDFORWARD): {DIM_FEEDFORWARD}
丢弃率 (DROPOUT): {DROPOUT}

四、目标变量
{'='*40}
{', '.join(TARGETS)}

Log10目标变换:
{'='*40}
LOG10_TARGETS: {', '.join(LOG10_TARGETS)}
LOG10_EPSILON_QUANTILE: {LOG10_EPSILON_QUANTILE}
LOG10_EPSILON_SCALE: {LOG10_EPSILON_SCALE}
HIGH_VALUE_QUANTILE: {HIGH_VALUE_QUANTILE}
HIGH_VALUE_LOSS_LAMBDA: {HIGH_VALUE_LOSS_LAMBDA}

五、测试结果详细指标
{'='*40}
"""

    # 添加每个目标的详细指标
    for target, metrics in global_metrics.items():
        report_content += f"\n{target}:\n"
        report_content += "-" * 30 + "\n"
        for metric_name, value in metrics.items():
            if metric_name == 'MRE(%)':
                report_content += f"  {metric_name}: {value:.4f}%\n"
            else:
                report_content += f"  {metric_name}: {value:.6e}\n"

    # 添加平均指标
    report_content += f"\n六、所有目标平均指标\n{'='*40}\n"
    for metric_name, value in avg_metrics.items():
        if metric_name == 'MRE(%)':
            report_content += f"平均{metric_name}: {value:.4f}%\n"
        else:
            report_content += f"平均{metric_name}: {value:.6e}\n"

    # 添加文件路径信息
    report_content += f"""
七、文件路径
{'='*40}
预训练模型路径: {PRETRAIN_MODEL_PATH}
微调模型路径: {FINETUNE_MODEL_PATH}
VGS曲线图路径: {FINE_PLOT_PATH}

八、数据统计
{'='*40}
训练文件数量: {len(TRAIN_FILES)}
验证文件数量: {len(VALID_FILES)}
测试文件数量: {len(TEST_FILES)}
目标变量数量: {len(TARGETS)}
"""

    # 保存报告
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n实验报告已保存至: {save_path}")
    return save_path


def save_predictions_to_csv(X_test, y_true, y_pred, targets, save_path, test_files=None):
    """
    将预测结果保存为 CSV。

    列结构 = 原始测试集 CSV 的列结构（列名、列顺序、其他列全部不变），
    仅把目标列的值替换为预测值。

    参数:
        X_test: 测试集输入特征（当 test_files 未提供时作为回退数据源）
        y_true: 真实目标值（保留接口兼容，不写入 CSV）
        y_pred: 预测目标值，形状 (N, target_dim) 或 (N, seq_len, target_dim)
        targets: 目标列名列表
        save_path: 保存路径
        test_files: 原始测试 CSV 路径列表（推荐传入，保证列结构完全一致）
    """
    # ---------- 统一 y_pred 维度为 (N, target_dim) ----------
    def _to_2d(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 3:
            # (N, seq_len, target_dim) -> 取最后一个时间步
            arr = arr[:, -1, :]
        return arr

    y_pred_arr = _to_2d(y_pred)

    if test_files is not None:
        # ---------- 直接读原始测试 CSV，保持列结构完全不变 ----------
        df_out = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)
        if len(df_out) != y_pred_arr.shape[0]:
            raise RuntimeError(
                f"原始测试CSV行数({len(df_out)}) 与预测值行数({y_pred_arr.shape[0]}) 不一致，"
                f"请检查 TEST_FILES 与数据加载顺序"
            )
        # 在原位置替换目标列的值为预测值（列名、顺序不变）
        for i, t in enumerate(targets):
            if i < y_pred_arr.shape[1]:
                if t not in df_out.columns:
                    raise KeyError(f"原始测试 CSV 中找不到目标列: {t}")
                df_out[t] = y_pred_arr[:, i]
    else:
        # ---------- 回退：基于 X_test 重建 ----------
        if isinstance(X_test, pd.DataFrame):
            df_out = X_test.copy().reset_index(drop=True)
        else:
            X_arr = np.asarray(X_test)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            df_out = pd.DataFrame(X_arr)
        for i, t in enumerate(targets):
            if i < y_pred_arr.shape[1]:
                df_out[t] = y_pred_arr[:, i]

    # ---------- 保存 ----------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_out.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存至: {save_path}")
    return df_out


def main():
    set_seed()
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE_ID)

    # 创建必要的目录
    create_directories([BASE_PLOT_PATH, FINE_PLOT_PATH, MODEL_SAVE_PATH])

    # ================== 数据加载 ==================
    print("Loading training data...")
    X_train, y_train_full, input_head, output_head = load_dataset(TRAIN_FILES)
    print("Loading validation data...")
    X_valid, y_valid_full, _, _ = load_dataset(VALID_FILES)
    print("Loading test data...")
    X_test, y_test_full, _, _ = load_dataset(TEST_FILES)

    # ================== 数据处理 ==================
    X_train, y_train, X_valid, y_valid, X_test, y_test = preprocess_data(
        X_train, y_train_full, X_valid, y_valid_full, X_test, y_test_full,
        input_head, output_head, TARGETS
    )

    X_train_scaled, X_valid_scaled, X_test_scaled, y_train_scaled, y_valid_scaled, y_test_scaled, scalers_y, scaler_X = scale_data(
        X_train, X_valid, X_test, y_train, y_valid, y_test, TARGETS
    )

    # FT 数据中留出 20% 仅用于轻量学生早停；测试集始终只做最终评估。
    split_rng = np.random.default_rng(42)
    split_indices = split_rng.permutation(len(X_valid_scaled))
    compact_valid_count = max(1, int(round(len(split_indices) * 0.2)))
    compact_valid_indices = split_indices[:compact_valid_count]
    finetune_indices = split_indices[compact_valid_count:]
    if len(finetune_indices) == 0:
        raise ValueError("VALID_FILES 样本过少，无法划分 FT 与轻量模型验证集")
    X_finetune_scaled = X_valid_scaled[finetune_indices]
    y_finetune_scaled = y_valid_scaled[finetune_indices]
    X_compact_valid_scaled = X_valid_scaled[compact_valid_indices]
    y_compact_valid_scaled = y_valid_scaled[compact_valid_indices]

    # ================== 模型定义 ==================
    model = MLPModel(
        input_dim=X_train_scaled.shape[1],
        target_dim=len(TARGETS),
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)

    # ================== 预训练阶段（改为直接加载已保存的预训练模型） ==================
    if not os.path.exists(PRETRAIN_MODEL_PATH):
        raise FileNotFoundError(f"❌ 预训练模型不存在: {PRETRAIN_MODEL_PATH}，请先完成预训练")

    print(f"\n跳过预训练，直接加载已保存的预训练模型: {PRETRAIN_MODEL_PATH}")
    pretrain_checkpoint = torch.load(
        PRETRAIN_MODEL_PATH, map_location=DEVICE, weights_only=False
    )

    # 兼容两种保存格式：state_dict 或 {'model_state_dict': ...}
    if isinstance(pretrain_checkpoint, dict) and 'model_state_dict' in pretrain_checkpoint:
        pretrain_state = pretrain_checkpoint['model_state_dict']
        best_pretrain_loss = pretrain_checkpoint.get('best_loss', float('nan'))
    else:
        pretrain_state = pretrain_checkpoint
        best_pretrain_loss = float('nan')

    output_weight = pretrain_state.get('output_layer.1.weight')
    expected_shape = (len(TARGETS), D_MODEL)
    if output_weight is None or tuple(output_weight.shape) != expected_shape:
        actual_shape = None if output_weight is None else tuple(output_weight.shape)
        raise RuntimeError(
            f"预训练权重输出层形状为 {actual_shape}，期望 {expected_shape}。"
            "请先在 script_MLP_Log_QS_VA 中运行 python main.py，"
            "生成 Qg/Qd/Qs/Ids 四输出预训练模型。"
        )
    model.load_state_dict(pretrain_state)

    print(f"✅ 预训练模型加载完成，best_pretrain_loss = {best_pretrain_loss}")

    # 由于跳过了预训练，loss 曲线用空列表占位
    pretrain_losses = []

    # ================== 微调阶段 ==================
    # 使用验证集作为微调数据
    finetune_losses, best_finetune_loss = finetune_model(
        model, X_finetune_scaled, y_finetune_scaled, TARGETS, DEVICE,
        BATCH_SIZE, FINETUNE_EPOCHS, LEARNING_RATE_FINETUNE,
        FINETUNE_MODEL_PATH, FINETUNE_PATIENCE,
        scalers_y=scalers_y, scaler_x=scaler_X
    )

    # 加载微调后的模型（早停机制已自动加载）
    print(f"\n加载微调模型，训练损失: {best_finetune_loss:.6f}")

    # 绘制损失曲线（只绘制微调损失，因为预训练阶段被跳过）
    print("\n绘制损失曲线...")
    plot_loss_curves(pretrain_losses, finetune_losses)

    # ================== 测试阶段 ==================
    y_true, y_pred = test_model(
        model, X_test_scaled, y_test_scaled, TARGETS, DEVICE,
        BATCH_SIZE, scalers_y
    )

    # ================== 保存预测结果到 CSV ==================
    print("\n保存预测结果到 CSV...")
    csv_save_path = os.path.join(BASE_PLOT_PATH, "predictions.csv")
    save_predictions_to_csv(
        X_test, y_true, y_pred, TARGETS, csv_save_path,
        test_files=TEST_FILES  # 传入原始测试 CSV，保证列结构完全一致
    )

    # ================== 结果可视化与评估 ==================
    print("\n生成散点图...")
    plot_scatter_comparison(y_true, y_pred, TARGETS, BASE_PLOT_PATH)

    print("\n生成VGS变化曲线图...")
    plot_vgs_comparison(X_test, y_true, y_pred, TARGETS, input_head, FINE_PLOT_PATH)

    print("\n全局评估指标:")
    global_metrics = calculate_metrics(y_true, y_pred, TARGETS)

    # 打印每个目标的指标
    for target, metrics in global_metrics.items():
        print(f"\n{target}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4e}" if k != 'MRE(%)' else f"{k}: {v:.2f}%")

    # 计算并打印平均指标
    avg_metrics = calculate_average_metrics(global_metrics)
    print("\n所有目标的平均指标:")
    for metric, value in avg_metrics.items():
        if metric != 'MRE(%)':
            print(f"平均{metric}: {value:.4e}")
        else:
            print(f"平均{metric}: {value:.2f}%")

    # ================== 保存实验报告 ==================
    print("\n保存实验报告...")
    report_path = os.path.join(MODEL_SAVE_PATH, "experiment_report.txt")
    save_experiment_report(global_metrics, avg_metrics, report_path)

    # ================== 完整微调模型直接导出为 Verilog-A ==================
    print("\n直接导出完整微调模型为 Verilog-A（不进行蒸馏）...")
    finetuned_va_path = export_finetuned_model(
        model=model,
        scaler_x=scaler_X,
        scalers_y=scalers_y,
        input_names=input_head,
        target_names=TARGETS,
        raw_domain_data=np.concatenate([X_train, X_valid], axis=0),
        output_path=FINETUNED_VA_PATH,
        module_name=FINETUNED_VA_MODULE,
    )

    print("\n=== 流程完成 ===")
    print(f"预训练模型(加载): {PRETRAIN_MODEL_PATH}")
    print(f"微调模型: {FINETUNE_MODEL_PATH}")
    print(f"测试结果已保存到: {BASE_PLOT_PATH}")
    print(f"预测结果 CSV: {csv_save_path}")
    print(f"实验报告已保存到: {report_path}")
    print(f"完整微调模型 Verilog-A: {finetuned_va_path}")


if __name__ == "__main__":
    main()
