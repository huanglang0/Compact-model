import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
import math
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# 只导入 BASE_PLOT_PATH
from config import BASE_PLOT_PATH


def plot_scatter_comparison(y_true, y_pred, target_names, save_path):
    """
    绘制 True vs Pred 散点图
    - 根据 target_names 数量动态计算网格行列，避免 3x3 不够用
    """
    n = len(target_names)
    if n == 0:
        print("⚠️ target_names 为空，跳过散点图绘制")
        return

    # 动态计算行列：优先接近正方形，也可以固定列数为 3
    ncols = 3
    nrows = int(math.ceil(n / ncols))

    plt.figure(figsize=(6 * ncols, 5 * nrows), dpi=100)

    for i, target in enumerate(target_names):
        # 动态 grid：subplot(nrows, ncols, i+1)
        plt.subplot(nrows, ncols, i + 1)

        col_idx = i  # 直接使用枚举索引，等价于 target_names.index(target)

        y_t = y_true[:, col_idx]
        y_p = y_pred[:, col_idx]

        # 计算统计指标
        r2 = r2_score(y_t, y_p)
        mse = mean_squared_error(y_t, y_p)

        # 绘制散点图
        plt.scatter(y_t, y_p, alpha=0.5, label=f'R²={r2:.4f}\nMSE={mse:.2e}')
        max_val = max(y_t.max(), y_p.max())
        min_val = min(y_t.min(), y_p.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

        # 设置坐标轴标签和标题
        plt.xlabel('True Value', fontsize=15)
        plt.ylabel('Predicted Value', fontsize=15)
        plt.title(target, fontsize=20)
        plt.legend(fontsize=14)
        plt.grid(True)

        # 增大刻度标签字体
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=18)

    # 关闭多余的子图（当 n < nrows*ncols 时）
    total_slots = nrows * ncols
    for j in range(n + 1, total_slots + 1):
        plt.subplot(nrows, ncols, j)
        plt.axis('off')

    plt.tight_layout()

    # 保存图表 - 仅保存为PNG
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # PNG版本
    scatter_png_path = os.path.join(save_path, "scatter_comparison.png")
    plt.savefig(scatter_png_path, bbox_inches='tight')
    print(f"散点图(PNG)已保存至: {scatter_png_path}")

    plt.show()
    plt.close()


def plot_vgs_comparison(X_test, y_true, y_pred, target_names, input_head, save_path):
    # 获取vgs和vds在输入特征中的索引
    vgs_idx = input_head.index('vgs') if 'vgs' in input_head else None
    vds_idx = input_head.index('vds') if 'vds' in input_head else None

    if vgs_idx is None or vds_idx is None:
        print("Warning: 'vgs' or 'vds' not found in input features, skipping VGS plots")
        return

    vgs_values = X_test[:, vgs_idx]
    vds_values = X_test[:, vds_idx]

    # 获取唯一的vds值
    unique_vds = np.unique(vds_values)
    print(f"Found {len(unique_vds)} unique VDS values: {unique_vds}")

    # ====== 自定义颜色方案 ======
    custom_colors = {
        0.5: ('red', 'blue'),
        1: ('green', 'black')
    }

    # 为其他Vds值创建颜色映射
    other_vds = [v for v in unique_vds if v not in custom_colors]
    if other_vds:
        norm = mcolors.Normalize(vmin=min(other_vds), vmax=max(other_vds))
        color_map = cm.viridis
    else:
        norm = None
        color_map = None

    # 为每个目标单独创建图表并保存
    for i, target in enumerate(target_names):
        plt.figure(figsize=(10, 8), dpi=100)

        col_idx = target_names.index(target)

        for vds in unique_vds:
            mask = vds_values == vds
            if np.sum(mask) == 0:
                continue

            vgs_current = vgs_values[mask]
            y_true_current = y_true[mask, col_idx]
            y_pred_current = y_pred[mask, col_idx]

            sort_idx = np.argsort(vgs_current)
            vgs_sorted = vgs_current[sort_idx]
            y_true_sorted = y_true_current[sort_idx]
            y_pred_sorted = y_pred_current[sort_idx]

            if vds in custom_colors:
                true_color, pred_color = custom_colors[vds]
            else:
                if norm and color_map:
                    color_val = norm(vds)
                    true_color = color_map(color_val)
                    pred_color = color_map(color_val)
                else:
                    true_color = 'blue'
                    pred_color = 'red'

            plt.plot(vgs_sorted, y_true_sorted, 'o-', markersize=4,
                     label=f'T_vds={vds:.2f}', color=true_color, alpha=0.7)
            plt.plot(vgs_sorted, y_pred_sorted, 'x-', markersize=4,
                     label=f'pre_vds={vds:.2f}', color=pred_color, alpha=0.7, linestyle='--')

        plt.xlabel('Vgs', fontsize=20)
        plt.ylabel(target, fontsize=20)
        plt.title(f'{target} vs Vgs', fontsize=20)
        plt.grid(True)

        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tick_params(axis='both', which='minor', labelsize=20)

        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle

        plt.legend(unique_labels.values(), unique_labels.keys(),
                   fontsize=14, loc='best', ncol=2)

        plt.tight_layout()

        os.makedirs(save_path, exist_ok=True)

        safe_target = target.replace(' ', '_').replace('/', '_')

        vgs_png_path = os.path.join(save_path, f"vgs_comparison_{safe_target}.png")
        plt.savefig(vgs_png_path, bbox_inches='tight')
        print(f"{target} VGS变化曲线图(PNG)已保存至: {vgs_png_path}")

        plt.close()


def plot_loss_curves(pretrain_losses, finetune_losses, pretrain_valid_losses=None, finetune_valid_losses=None):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(14, 6))

    # 预训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(pretrain_losses, 'b-', linewidth=2, label='训练损失')
    if pretrain_valid_losses is not None:
        plt.plot(pretrain_valid_losses, 'r-', linewidth=2, label='验证损失')
    plt.title('预训练损失曲线', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # 微调损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(finetune_losses, 'b-', linewidth=2, label='训练损失')
    if finetune_valid_losses is not None:
        plt.plot(finetune_valid_losses, 'r-', linewidth=2, label='验证损失')
    plt.title('微调损失曲线', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    os.makedirs(BASE_PLOT_PATH, exist_ok=True)
    loss_curve_path = os.path.join(BASE_PLOT_PATH, "loss_curves.png")
    plt.savefig(loss_curve_path, bbox_inches='tight', dpi=300)
    print(f"损失曲线图已保存至: {loss_curve_path}")

    plt.show()
    plt.close()