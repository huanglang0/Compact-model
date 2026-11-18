#评估指标
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred, target_names):
    metrics = {}
    for i, target in enumerate(target_names):
        # 通过目标名称获取列索引
        col_idx = target_names.index(target)
        y_t = y_true[:, col_idx]
        y_p = y_pred[:, col_idx]

        # 计算指标
        mse = mean_squared_error(y_t, y_p)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_t, y_p)

        # MRE使用全部样本；真实值为0时只做除零保护，不过滤样本
        denominator = np.where(y_t == 0, np.finfo(np.float64).eps, y_t)
        mre = np.mean(np.abs((y_t - y_p) / denominator)) * 100

        metrics[target] = {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MRE(%)': mre
        }
    return metrics


def calculate_average_metrics(metrics_dict):
    """计算所有目标的平均指标"""
    avg_metrics = {
        'MSE': 0.0,
        'RMSE': 0.0,
        'R2': 0.0,
        'MRE(%)': 0.0
    }
    valid_targets = 0

    for target, metrics in metrics_dict.items():
        for k, v in metrics.items():
            # 跳过NaN值
            if not np.isnan(v):
                avg_metrics[k] += v
                # 只对有效目标计数
                if k == 'MSE':
                    valid_targets += 1

    # 计算平均值
    if valid_targets > 0:
        for k in avg_metrics:
            avg_metrics[k] /= valid_targets

    return avg_metrics
