# 整合所有组件的主流程：MLP一次性输出所有目标，没有自回归，没有评估阈值过滤
import os
import torch
import numpy as np
from config import *
from model import MLPModel
from data_processing import load_dataset, preprocess_data, scale_data
from train import pretrain_model, finetune_model, test_model
from metrics import calculate_metrics, calculate_average_metrics
from visualization import plot_scatter_comparison, plot_vgs_comparison, plot_loss_curves
from compact_distill import train_compact_student
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

    # FT 数据中留出 20% 仅用于轻量学生模型的早停，避免用测试集选模型。
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

    # ================== 预训练阶段 ==================
    # 方法1: 直接传递 patience 值
    pretrain_losses, best_pretrain_loss = pretrain_model(
        model, X_train_scaled, y_train_scaled, TARGETS, DEVICE,
        BATCH_SIZE, PRETRAIN_EPOCHS, LEARNING_RATE_PRETRAIN, WEIGHT_DECAY,
        PRETRAIN_MODEL_PATH, PRETRAIN_PATIENCE, scalers_y  # 直接传递值，不使用关键字参数
    )

    # 或者方法2: 如果函数没有 patience 参数，从 config 中移除并硬编码
    # pretrain_losses, best_pretrain_loss = pretrain_model(
    #     model, X_train_scaled, y_train_scaled, TARGETS, DEVICE,
    #     BATCH_SIZE, PRETRAIN_EPOCHS, LEARNING_RATE_PRETRAIN, WEIGHT_DECAY,
    #     PRETRAIN_MODEL_PATH
    # )

    # 加载最佳预训练模型（早停机制已自动加载）
    print(f"\n加载最佳预训练模型，训练损失: {best_pretrain_loss:.6f}")

    # ================== 微调阶段 ==================
    # 使用验证集作为微调数据
    # 方法1: 直接传递 patience 值
    finetune_losses, best_finetune_loss = finetune_model(
        model, X_finetune_scaled, y_finetune_scaled, TARGETS, DEVICE,
        BATCH_SIZE, FINETUNE_EPOCHS, LEARNING_RATE_FINETUNE,
        FINETUNE_MODEL_PATH, FINETUNE_PATIENCE, scalers_y  # 直接传递值，不使用关键字参数
    )

    # 或者方法2: 如果函数没有 patience 参数，从 config 中移除并硬编码
    # finetune_losses, best_finetune_loss = finetune_model(
    #     model, X_valid_scaled, y_valid_scaled, TARGETS, DEVICE,
    #     BATCH_SIZE, FINETUNE_EPOCHS, LEARNING_RATE_FINETUNE,
    #     FINETUNE_MODEL_PATH
    # )

    # 加载微调后的模型（早停机制已自动加载）
    print(f"\n加载微调模型，训练损失: {best_finetune_loss:.6f}")

    # 绘制损失曲线（只绘制训练损失）
    print("\n绘制损失曲线...")
    plot_loss_curves(pretrain_losses, finetune_losses)

    # ================== 测试阶段 ==================
    # 使用并行计算进行测试（提供目标值但不用于教师强制）
    y_true, y_pred = test_model(
        model, X_test_scaled, y_test_scaled, TARGETS, DEVICE,
        BATCH_SIZE, scalers_y
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

    # ================== 轻量学生蒸馏与 Verilog-A 导出 ==================
    print("\n训练轻量准静态学生模型并导出 Verilog-A...")
    compact_model, compact_bundle_path, compact_va_path, compact_valid_loss = train_compact_student(
        teacher=model,
        x_pretrain_scaled=X_train_scaled,
        y_pretrain_scaled=y_train_scaled,
        x_ft_scaled=X_finetune_scaled,
        y_ft_scaled=y_finetune_scaled,
        x_valid_scaled=X_compact_valid_scaled,
        y_valid_scaled=y_compact_valid_scaled,
        scaler_x=scaler_X,
        scalers_y=scalers_y,
        input_names=input_head,
        target_names=TARGETS,
        raw_domain_data=np.concatenate([X_train, X_valid], axis=0),
    )
    print(f"轻量学生验证集 scaled MSE: {compact_valid_loss:.6e}")

    # 独立测试集上评估最终会被导出的学生模型，而不仅是教师模型。
    compact_model.eval()
    compact_pred_scaled = np.zeros_like(y_test_scaled)
    with torch.no_grad():
        for batch_start in range(0, len(X_test_scaled), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(X_test_scaled))
            compact_input = torch.tensor(
                X_test_scaled[batch_start:batch_end], dtype=torch.float32, device=DEVICE
            )
            compact_pred_scaled[batch_start:batch_end] = compact_model(compact_input).cpu().numpy()
    compact_y_pred = np.zeros_like(compact_pred_scaled)
    for index, scaler in enumerate(scalers_y):
        compact_y_pred[:, index] = scaler.inverse_transform(
            compact_pred_scaled[:, index:index + 1]
        ).ravel()
    compact_metrics = calculate_metrics(y_true, compact_y_pred, TARGETS)
    compact_avg_metrics = calculate_average_metrics(compact_metrics)
    compact_report_path = os.path.join(MODEL_SAVE_PATH, "compact_qs_experiment_report.txt")
    save_experiment_report(compact_metrics, compact_avg_metrics, compact_report_path)

    print("\n=== 流程完成 ===")
    print(f"预训练模型: {PRETRAIN_MODEL_PATH}")
    print(f"微调模型: {FINETUNE_MODEL_PATH}")
    print(f"测试结果已保存到: {BASE_PLOT_PATH}")
    print(f"实验报告已保存到: {report_path}")
    print(f"轻量部署包: {compact_bundle_path}")
    print(f"准静态 Verilog-A: {compact_va_path}")
    print(f"轻量模型测试报告: {compact_report_path}")


if __name__ == "__main__":
    main()
