# CLE-Net — 共聚焦激光显微内镜(CLE)图像分类网络

从 `/home/u2518483080/Classification` 抽取的 CLE-Net 完整自洽副本(2026-10-26)。

## 网络结构(核心文件)

| 文件 | 内容 |
|---|---|
| `models/optimize/convnext_cle2.py` | **CLE-Net 最终形态** `convnext_egb_scg_hsc`: EGB(PFESA 零参数边缘-结构门控) + SCG(条带上下文引导) + HSC(通道自注意力校准头); 另含 egb/scg/hsc/egb_scg/scg_hsc 消融变体 |
| `models/Convnext.py` | ConvNeXt-T baseline 主干(CLE-Net 的父类与权重来源) |
| `models/optimize/convnext_cle3~5.py` | v2~v4 探索版本(OS/DSH/PLD/ESA), 保留完整实验演进记录 |

## 依赖关系

```
convnext_cle2.py ── from models.Convnext import ConvNeXt, Block
```
即: CLE-Net(结构) = models/optimize/convnext_cle2.py + models/Convnext.py, 仅此两个文件。

## 训练与评测

```bash
# 训练(300 epochs, early-stop 30, ImageNet ema 预训练)
python train_cle_fair.py --model convnext_egb_scg_hsc --num_classes 3 --img_size 224 \
  --seed 42 --data-path-train ./datasets/cle_ac_v1/train --data-path-val ./datasets/cle_ac_v1/val \
  --weights <convnext_tiny_1k_224_ema.pth 路径> --save-dir ./weights/cle_net_seed42 --epochs 300

# 评测(全指标: macro/weighted F1, balanced acc, AUROC, AUC-PR, kappa, MCC, per-class sens)
python evaluate.py --model convnext_egb_scg_hsc \
  --weights ./weights/cle_net_seed42/convnext_egb_scg_hsc/model_best.pth --data-path ./datasets/cle_ac_v1/test
```

配套文件: `train_cle_fair.py`(公平训练协议) / `evaluate.py`(评测) / `cle_fair_common.py`(模型注册表) /
`my_dataset.py`(数据集) / `utils.py`(数据读取) / `consensus_manifest.py`(样本清单)。

注意: 数据集与预训练权重不在本副本内(体积大), 位置见 experiments/FINAL_REPORT_n1.md。

## 主要结果(test 集, best-of-3-seeds)

| 网络 | Macro-F1 | Acc | AUROC |
|---|---|---|---|
| **CLE-Net (egb_scg_hsc)** | **0.9667** | **0.9670** | **0.9974** |
| ConvNeXt-T baseline | 0.9633 | 0.9635 | 0.9967 |

详见 `experiments/COMPARE_TABLE.md`(10 网络对比) 与 `experiments/FINAL_REPORT_n1.md`(完整实验报告)。
