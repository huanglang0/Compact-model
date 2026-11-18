# 四输出预训练模型的 FT、测试与 Verilog-A 导出

本目录不重复预训练。它与相邻的 `script_MLP_Log_QS_VA` 配套使用：

```text
script_MLP_Log_QS_VA
  四输出预训练 → saved_models/best_pretrain_model.pth
                         ↓
script_MLP_Log_FT_QS_VA
  加载预训练权重 → FT → 测试 → 轻量学生蒸馏 → 导出 .va
```

两边必须使用完全相同的模型结构、输入顺序、训练数据归一化方式和输出顺序：

```text
QG_TOTAL, QD_TOTAL, QS_TOTAL, ids
```

## 运行顺序

先运行预训练工程：

```bash
cd ../script_MLP_Log_QS_VA
python pretrain_only.py
```

确认生成：

```text
script_MLP_Log_QS_VA/saved_models/best_pretrain_model.pth
```

再运行本 FT 工程：

```bash
cd ../script_MLP_Log_FT_QS_VA
python main.py
```

本目录最终生成：

- `saved_models/best_finetune_model.pth`
- `saved_models/compact_qs_bundle.pth`
- `saved_models/compact_qs_finfet.va`
- `saved_models/compact_qs_experiment_report.txt`

`compact_qs_finfet.va` 使用静态 `Ids` 完成 DC，通过 `ddt(Qg/Qd/Qb)` 支持准静态
AC 和瞬态，并用 `Qb = -(Qg + Qd + Qs)` 保证终端电荷守恒。最终 VA 只包含
`6→24→24→4` 的 868 参数轻量学生，不包含大教师网络。
