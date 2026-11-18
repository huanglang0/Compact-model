#配置参数
import os
import torch
from pathlib import Path

# 设备配置
DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
if torch.cuda.is_available() and DEVICE_ID < torch.cuda.device_count():
    torch.cuda.set_device(DEVICE_ID)
    DEVICE = torch.device(f"cuda:{DEVICE_ID}")
else:
    DEVICE = torch.device("cpu")

# 训练参数
BATCH_SIZE = 512
PRETRAIN_EPOCHS = 500
FINETUNE_EPOCHS = 2000
LEARNING_RATE_PRETRAIN = 0.0008
LEARNING_RATE_FINETUNE = 0.0001
WEIGHT_DECAY = 1e-4
# Apply Ids(Vds=0)=0 during fine-tuning instead of evaluating the full
# network twice inside Verilog-A.
FINETUNE_ZERO_VDS_LAMBDA = 0.1

# Early Stopping 参数
PRETRAIN_PATIENCE = 10  # 预训练早停耐心值
FINETUNE_PATIENCE = 200  # 微调早停耐心值
DELTA = 0          # 改进阈值

# 模型参数
D_MODEL = 64
NHEAD = 8
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0

# 必须与 script_MLP_Log_QS_VA 的预训练目标和顺序完全一致。
TARGETS = ['QG_TOTAL', 'QD_TOTAL', 'QS_TOTAL', 'ids']

# 只对这些动态范围很大的目标做 signed-log10 目标变换。
# 其他目标仍保持原来的 StandardScaler 处理方式。
LOG10_TARGETS = ['ids']

LOG10_EPSILON_QUANTILE = 0.1
LOG10_EPSILON_SCALE = 0.1
HIGH_VALUE_QUANTILE = 0.75
HIGH_VALUE_LOSS_LAMBDA = 0.1

def _csv_paths(name: str):
    """Read a semicolon-separated list of CSV paths from an environment variable."""
    value = os.getenv(name, "")
    paths = [item.strip() for item in value.split(";") if item.strip()]
    if not paths:
        raise RuntimeError(
            f"{name} is not set. Set it to one or more CSV paths separated by ';'. "
            "See README.md for an example."
        )
    return paths


# Dataset locations are deliberately supplied at runtime so machine-specific paths
# and potentially proprietary simulation data are never committed to Git.
TRAIN_FILES = _csv_paths("TRAIN_FILES")
VALID_FILES = _csv_paths("VALID_FILES")
TEST_FILES = _csv_paths("TEST_FILES")

# 路径配置：预训练权重来自相邻的预训练工程；FT产物保存在本目录。
PROJECT_DIR = Path(__file__).resolve().parent
BASE_PLOT_PATH = str(PROJECT_DIR / "picture")
FINE_PLOT_PATH = str(PROJECT_DIR / "picture")
MODEL_SAVE_PATH = str(PROJECT_DIR / "saved_models")
PRETRAIN_MODEL_PATH = os.getenv(
    "PRETRAIN_MODEL_PATH", str(Path(MODEL_SAVE_PATH) / "best_pretrain_model.pth")
)
FINETUNE_MODEL_PATH = f"{MODEL_SAVE_PATH}/best_finetune_model.pth"
REPORT_SAVE_PATH = f"{MODEL_SAVE_PATH}/experiment_report.txt"  # 实验报告保存路径

# 直接部署微调后的完整教师模型，不经过学生模型蒸馏。
FINETUNED_VA_PATH = f"{MODEL_SAVE_PATH}/finetuned_qs_finfet_nmos.va"
FINETUNED_VA_MODULE = "finetuned_qs_finfet_nmos"

# 准静态 Verilog-A 部署配置。原 FT 教师模型保持不变；以下参数只用于轻量学生模型。
QS_TARGETS = TARGETS
COMPACT_HIDDEN_DIMS = (64, 64)
COMPACT_EPOCHS = 1200
COMPACT_BATCH_SIZE = 512
COMPACT_LEARNING_RATE = 5e-4
COMPACT_WEIGHT_DECAY = 1e-6
COMPACT_PATIENCE = 80
COMPACT_BASE_TO_FT_RATIO = 3
COMPACT_DISTILL_LAMBDA = 0.35
COMPACT_DERIVATIVE_LAMBDA = 0.05
COMPACT_ZERO_VDS_LAMBDA = 0.02
COMPACT_OUTPUT_DIR = Path(__file__).resolve().parent / "saved_models"
COMPACT_BUNDLE_PATH = str(COMPACT_OUTPUT_DIR / "compact_qs_bundle.pth")
COMPACT_VA_PATH = str(COMPACT_OUTPUT_DIR / "compact_qs_finfet.va")
