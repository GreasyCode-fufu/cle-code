# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_abl/convnext_v2c_seed44/convnext_v2c/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8163 |
| Accuracy | 0.8473 |
| Macro-AUROC (OvR) | 0.9621 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8466 |
| Balanced Acc | 0.8224 |
| Macro-Sensitivity | 0.8224 |
| Macro-Specificity | 0.9687 |
| Macro-Precision | 0.8168 |
| AUC-PR (Macro) | 0.8747 |
| Cohen's Kappa | 0.8040 |
| MCC | 0.8051 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.26M |
| Pure-Inference FPS | 1796.4 (0.56 ms/张) |
| End-to-End FPS | 490.5 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8945 | 0.9959 | 0.8054 | 0.8476 | 0.9980 | 0.9982 | 0.9311 | 199 |
| cancer | 0.8922 | 0.9708 | 0.9101 | 0.9011 | 0.9646 | 0.9782 | 0.9569 | 2645 |
| empty | 0.9957 | 0.9637 | 0.9236 | 0.9583 | 0.9980 | 0.9851 | 0.9645 | 3256 |
| normal | 0.7274 | 0.9899 | 0.9334 | 0.8176 | 0.9490 | 0.9556 | 0.8904 | 1735 |
| other | 0.6715 | 0.9368 | 0.6715 | 0.6715 | 0.9368 | 0.8939 | 0.7446 | 1717 |
| stroma | 0.7532 | 0.9554 | 0.6567 | 0.7017 | 0.9716 | 0.9615 | 0.7604 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=178 FP=43 TN=10392 FN=21
- **cancer**: TP=2360 FP=233 TN=7756 FN=285
- **empty**: TP=3242 FP=268 TN=7110 FN=14
- **normal**: TP=1262 FP=90 TN=8809 FN=473
- **other**: TP=1153 FP=564 TN=8353 FN=564
- **stroma**: TP=815 FP=426 TN=9126 FN=267

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 178 | 5 | 2 | 4 | 9 | 1 |
| cancer | 3 | 2360 | 0 | 19 | 208 | 55 |
| empty | 1 | 2 | 3242 | 0 | 11 | 0 |
| normal | 1 | 95 | 0 | 1262 | 163 | 214 |
| other | 33 | 92 | 266 | 17 | 1153 | 156 |
| stroma | 5 | 39 | 0 | 50 | 173 | 815 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8054    0.8945    0.8476       199
      cancer     0.9101    0.8922    0.9011      2645
       empty     0.9236    0.9957    0.9583      3256
      normal     0.9334    0.7274    0.8176      1735
       other     0.6715    0.6715    0.6715      1717
      stroma     0.6567    0.7532    0.7017      1082

    accuracy                         0.8473     10634
   macro avg     0.8168    0.8224    0.8163     10634
weighted avg     0.8518    0.8473    0.8466     10634

```