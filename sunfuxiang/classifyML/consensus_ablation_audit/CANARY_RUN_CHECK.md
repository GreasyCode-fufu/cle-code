# CLE 共识消融 Canary Run 检查

检查时间：2026-09-16（Asia/Shanghai）  
Run：`subset=R, seed=42, model=ResNet34`  
结论：`CANARY_RUN_CHECK = PASS`

## 训练与评估

- manifest：`manifests/consensus_ablation/R.csv`
- manifest SHA-256：`aba7178d193f58c19cbb2c1ca00714f6d10bd2fb5fbf0aa390035201c9c55e96`
- 实际训练样本：3500（Normal 1405 / Benign 1458 / Malignant 637）
- 固定 validation：538
- 固定 test：671
- 训练 epoch：62（epoch 0–61）
- 早停：epoch 61；最佳 validation Macro-F1 为 0.8839（epoch 31）
- test Macro-F1：0.8792762454
- test Accuracy：0.8792846498
- test Macro Recall：0.8807246298
- test Macro Precision：0.8808628893
- test Weighted F1：0.8776560264
- test Macro AUROC OvR：0.9758469845

## 必检项

| 检查项 | 结果 | 证据 |
|---|---|---|
| best checkpoint 存在且有效 | PASS | `resnet34_R_seed42_best.pth`，SHA-256 `cf35220f...a65a24` |
| train log 完整 | PASS | 62 行 epoch 记录；SHA-256 `2e681f7c...98dcaf` |
| run_meta 完整 | PASS | subset/seed/manifest hash/样本与三类计数/优化器/lr/weight decay/scheduler/early stop/pretrained hash/code hash/timestamp 全部校验 |
| predictions.csv 恰好 671 行 | PASS | 无重复 image_path；SHA-256 `e87022df...2faa3` |
| probability 列正常 | PASS | 3 列；全部为有限 `[0,1]` 值；逐行和为 1（容差 `2e-5`） |
| 统一汇总逻辑可重算 Macro-F1 | PASS | 重算值与 `results.json` 一致：`0.8792762454065937` |
| Full/HQ 旧文件未修改 | PASS | Canary 启动后六个旧权重目录新增/修改文件数为 0；六份锁定 predictions SHA-256 全部匹配 |
| val/test 未变化 | PASS | val=538、test=671；与 highquality 固定副本目录一致；Canary 启动后修改文件数为 0 |

## 协议核验

`run_meta.json` 记录并通过严格检查：ResNet34、输入 224、batch 32、max epochs 300、AdamW、lr 1e-4、weight decay 0.05、CosineAnnealingLR(T_max=300, eta_min=0)、unweighted cross-entropy、label smoothing 0、AMP、early stopping patience 30、按 validation Macro-F1 选优、预训练权重 SHA-256 `333f7ec4...c6cef`。

Canary 全部通过，可以进入剩余 17 个 run 的安全串行训练。Random-2880 未启动。
