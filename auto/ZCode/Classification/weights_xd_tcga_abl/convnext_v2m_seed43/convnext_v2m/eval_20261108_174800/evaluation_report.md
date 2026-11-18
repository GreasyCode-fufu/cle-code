# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_abl/convnext_v2m_seed43/convnext_v2m/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8227 |
| Accuracy | 0.8583 |
| Macro-AUROC (OvR) | 0.9736 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8527 |
| Balanced Acc | 0.8139 |
| Macro-Sensitivity | 0.8139 |
| Macro-Specificity | 0.9698 |
| Macro-Precision | 0.8401 |
| AUC-PR (Macro) | 0.8973 |
| Cohen's Kappa | 0.8166 |
| MCC | 0.8189 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 603.2 (1.66 ms/张) |
| End-to-End FPS | 315.2 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8291 | 0.9962 | 0.8049 | 0.8168 | 0.9967 | 0.9977 | 0.9054 | 199 |
| cancer | 0.9614 | 0.9279 | 0.8153 | 0.8824 | 0.9864 | 0.9871 | 0.9647 | 2645 |
| empty | 0.9926 | 0.9629 | 0.9218 | 0.9559 | 0.9966 | 0.9859 | 0.9676 | 3256 |
| normal | 0.8115 | 0.9870 | 0.9239 | 0.8641 | 0.9641 | 0.9768 | 0.9312 | 1735 |
| other | 0.6057 | 0.9620 | 0.7542 | 0.6718 | 0.9269 | 0.9235 | 0.7751 | 1717 |
| stroma | 0.6830 | 0.9830 | 0.8202 | 0.7453 | 0.9648 | 0.9704 | 0.8401 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=165 FP=40 TN=10395 FN=34
- **cancer**: TP=2543 FP=576 TN=7413 FN=102
- **empty**: TP=3232 FP=274 TN=7104 FN=24
- **normal**: TP=1408 FP=116 TN=8783 FN=327
- **other**: TP=1040 FP=339 TN=8578 FN=677
- **stroma**: TP=739 FP=162 TN=9390 FN=343

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 165 | 9 | 2 | 6 | 16 | 1 |
| cancer | 5 | 2543 | 0 | 24 | 51 | 22 |
| empty | 0 | 0 | 3232 | 0 | 24 | 0 |
| normal | 1 | 174 | 0 | 1408 | 73 | 79 |
| other | 32 | 275 | 272 | 38 | 1040 | 60 |
| stroma | 2 | 118 | 0 | 48 | 175 | 739 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8049    0.8291    0.8168       199
      cancer     0.8153    0.9614    0.8824      2645
       empty     0.9218    0.9926    0.9559      3256
      normal     0.9239    0.8115    0.8641      1735
       other     0.7542    0.6057    0.6718      1717
      stroma     0.8202    0.6830    0.7453      1082

    accuracy                         0.8583     10634
   macro avg     0.8401    0.8139    0.8227     10634
weighted avg     0.8561    0.8583    0.8527     10634

```