# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce16_seed43/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7852 |
| Accuracy | 0.8257 |
| Macro-AUROC (OvR) | 0.9396 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8209 |
| Balanced Acc | 0.7855 |
| Macro-Sensitivity | 0.7855 |
| Macro-Specificity | 0.9637 |
| Macro-Precision | 0.7904 |
| AUC-PR (Macro) | 0.8333 |
| Cohen's Kappa | 0.7752 |
| MCC | 0.7765 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9617.6 (0.10 ms/张) |
| End-to-End FPS | 554.2 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8392 | 0.9954 | 0.7767 | 0.8068 | 0.9969 | 0.9948 | 0.9013 | 199 |
| cancer | 0.9089 | 0.9369 | 0.8267 | 0.8658 | 0.9688 | 0.9771 | 0.9290 | 2645 |
| empty | 0.9942 | 0.9612 | 0.9188 | 0.9550 | 0.9973 | 0.9822 | 0.9353 | 3256 |
| normal | 0.7193 | 0.9813 | 0.8826 | 0.7926 | 0.9472 | 0.9135 | 0.8500 | 1735 |
| other | 0.5824 | 0.9462 | 0.6757 | 0.6256 | 0.9217 | 0.8378 | 0.6801 | 1717 |
| stroma | 0.6691 | 0.9613 | 0.6618 | 0.6654 | 0.9625 | 0.9322 | 0.7040 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=167 FP=48 TN=10387 FN=32
- **cancer**: TP=2404 FP=504 TN=7485 FN=241
- **empty**: TP=3237 FP=286 TN=7092 FN=19
- **normal**: TP=1248 FP=166 TN=8733 FN=487
- **other**: TP=1000 FP=480 TN=8437 FN=717
- **stroma**: TP=724 FP=370 TN=9182 FN=358

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 167 | 6 | 2 | 8 | 13 | 3 |
| cancer | 6 | 2404 | 1 | 49 | 104 | 81 |
| empty | 0 | 0 | 3237 | 0 | 19 | 0 |
| normal | 0 | 187 | 0 | 1248 | 175 | 125 |
| other | 39 | 209 | 283 | 25 | 1000 | 161 |
| stroma | 3 | 102 | 0 | 84 | 169 | 724 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7767    0.8392    0.8068       199
      cancer     0.8267    0.9089    0.8658      2645
       empty     0.9188    0.9942    0.9550      3256
      normal     0.8826    0.7193    0.7926      1735
       other     0.6757    0.5824    0.6256      1717
      stroma     0.6618    0.6691    0.6654      1082

    accuracy                         0.8257     10634
   macro avg     0.7904    0.7855    0.7852     10634
weighted avg     0.8219    0.8257    0.8209     10634

```