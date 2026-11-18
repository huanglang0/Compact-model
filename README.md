# FinFET MLP QS / Verilog-A

Use a multilayer perceptron (MLP) to model four FinFET outputs—`QG_TOTAL`, `QD_TOTAL`, `QS_TOTAL`, and `ids`. The project supports pre-training, fine-tuning, metric/plot generation, compact-student distillation, and Verilog-A export.

## What is included

- `main_pretrain_ft.py`: train the teacher model, fine-tune it, then distill and export the compact quasi-static model.
- `main_onlyft.py`: load an existing pre-trained model, fine-tune it, evaluate it, and export the full teacher as Verilog-A.
- `export_compact_qs_va.py`: export a saved compact-student bundle as Verilog-A.
- `compare_csv_derivatives.py`: compare derivatives in reference and predicted CSV files.

## Requirements

Python 3.10+ is recommended. Install a PyTorch build appropriate for your CPU/CUDA environment, then install the remaining packages:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset configuration

Simulation datasets and trained weights are intentionally excluded from Git. Each CSV must contain these columns:

```text
vgs, vds, vbs, length, nfin, temp, QG_TOTAL, QD_TOTAL, QS_TOTAL, ids
```

Before running, set the three dataset variables. Multiple CSV files can be separated with `;`.

```powershell
$env:TRAIN_FILES = 'F:\data\train.csv'
$env:VALID_FILES = 'F:\data\valid.csv'
$env:TEST_FILES  = 'F:\data\test.csv'
python main_pretrain_ft.py
```

On Linux/macOS, replace the environment-variable commands with `export TRAIN_FILES=/path/to/train.csv`, and so on.

For fine-tuning from a previous checkpoint, point `PRETRAIN_MODEL_PATH` to it and run:

```powershell
$env:PRETRAIN_MODEL_PATH = 'F:\models\best_pretrain_model.pth'
python main_onlyft.py
```

`DEVICE_ID` is optional and defaults to `0`; use it to choose a CUDA GPU. The program falls back to CPU when CUDA is unavailable or that device is not present.

## Outputs

All generated files are written locally and ignored by Git:

- `saved_models/`: checkpoints, reports, and exported Verilog-A
- `picture/`: predictions and plots

## Notes

The model uses signed-log10 scaling for `ids`. Keep the input/output column names and their order consistent between all train, validation, and test files. `README_original.md` preserves the source project's original usage notes.
