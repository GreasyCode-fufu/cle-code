# CLE-Net — 共聚焦激光显微内镜(CLE)图像分类网络（审计后最终版）

自洽副本(2026-10-26 抽取)；**2026-11-03 按修复审计更新为最终版**：
最终形态由 `convnext_egb_scg_hsc` 更正为 **`convnext_egb_scg`**（HSC 经修复版
审计否决，组合贡献 −0.64pp，仅保留于消融表；TCGA 域例外见下）。

## 网络结构(核心文件)

| 文件 | 内容 |
|---|---|
| `models/optimize/convnext_cle2.py` | **CLE-Net 最终形态** `convnext_egb_scg`: EGB(PFESA 零参数边缘-结构门控, stage1-3) + SCG(条带上下文引导, stage3/4); 另含 egb/scg/hsc/egb_scg_hsc 等消融变体(hsc 版仅用于消融复现) |
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
python train_cle_fair.py --model convnext_egb_scg --num_classes 3 --img_size 224 \
  --seed 42 --data-path-train ./datasets/cle_ac_v1/train --data-path-val ./datasets/cle_ac_v1/val \
  --weights <convnext_tiny_1k_224_ema.pth 路径> --save-dir ./weights/cle_net_seed42 --epochs 300

# 评测(全指标: macro/weighted F1, balanced acc, AUROC, AUC-PR, kappa, MCC, per-class sens)
python evaluate.py --model convnext_egb_scg \
  --weights ./weights/cle_net_seed42/convnext_egb_scg/model_best.pth --data-path ./datasets/cle_ac_v1/test
```

配套文件: `train_cle_fair.py`(公平训练协议) / `evaluate.py`(评测) / `cle_fair_common.py`(模型注册表) /
`my_dataset.py`(数据集) / `utils.py`(数据读取) / `consensus_manifest.py`(样本清单)。

注意: 数据集与预训练权重不在本副本内(体积大), 位置见 experiments/FINAL_REPORT_n1.md。

## 主要结果

### CLE 主域(test 576 张, 修复版 3 seeds 42/43/44)

| 网络 | Macro-F1 均值±std | best | 配对Δ |
|---|---|---|---|
| **CLE-Net (egb_scg)** | **0.9630±0.0053** | 0.9688 | **+1.41pp (3/3)** |
| ConvNeXt-T baseline | 0.9489±0.0122 | 0.9615 | — |

修复审计前的全部组合数字(含旧文档的 0.9667 best-of-3)已作废, 不得引用。

### 跨域(vs 各域 convnext_tiny, 审计表06)

covid **+0.53pp(3/3)** / busi +0.22pp(5/6) / cub +0.32pp(2/3) / 
**TCGA(2026-11 新)**: egb_scg −0.36pp, hsc 版 +1.14pp(域特异例外, 两版并报)。

主干横评(12 模型×3 泛化数据集): CLE-Net 全部位列前二。

## 实验数据(experiment_data/, 2026-11-03 分发)

`metrics_clenet_all_datasets.csv`(各数据集逐种子+均值) ·
`backbone_comparison_cle.csv`(CLE 主域 18 网络) ·
`backbone_comparison_crossdomain.csv`(10 主干×3 域) ·
`tcga_all_networks.csv`(TCGA 全域对比)。
权威审计数据源: `Network/CLE-Net/experiments/sci_paper_data/`(14 张表)。

详见 `创新说明.md`(审计后版) 与 Network/CLE-Net 的 `experiments/SCI_DATA_REPORT.md`。
