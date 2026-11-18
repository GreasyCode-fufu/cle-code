# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/pvt_tiny_seed44/pvt_tiny/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7880 |
| Accuracy | 0.8320 |
| Macro-AUROC (OvR) | 0.9594 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8289 |
| Balanced Acc | 0.7843 |
| Macro-Sensitivity | 0.7843 |
| Macro-Specificity | 0.9655 |
| Macro-Precision | 0.7970 |
| AUC-PR (Macro) | 0.8563 |
| Cohen's Kappa | 0.7837 |
| MCC | 0.7846 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 1153.1 (0.87 ms/张) |
| End-to-End FPS | 421.9 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8442 | 0.9965 | 0.8195 | 0.8317 | 0.9970 | 0.9955 | 0.9123 | 199 |
| cancer | 0.8862 | 0.9772 | 0.9279 | 0.9066 | 0.9629 | 0.9843 | 0.9655 | 2645 |
| empty | 0.9972 | 0.9618 | 0.9201 | 0.9571 | 0.9987 | 0.9819 | 0.9491 | 3256 |
| normal | 0.7850 | 0.9651 | 0.8141 | 0.7993 | 0.9584 | 0.9729 | 0.8931 | 1735 |
| other | 0.6849 | 0.9199 | 0.6222 | 0.6521 | 0.9381 | 0.8969 | 0.7447 | 1717 |
| stroma | 0.5083 | 0.9727 | 0.6782 | 0.5811 | 0.9458 | 0.9249 | 0.6733 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=168 FP=37 TN=10398 FN=31
- **cancer**: TP=2344 FP=182 TN=7807 FN=301
- **empty**: TP=3247 FP=282 TN=7096 FN=9
- **normal**: TP=1362 FP=311 TN=8588 FN=373
- **other**: TP=1176 FP=714 TN=8203 FN=541
- **stroma**: TP=550 FP=261 TN=9291 FN=532

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 168 | 5 | 2 | 2 | 22 | 0 |
| cancer | 2 | 2344 | 4 | 92 | 95 | 108 |
| empty | 0 | 2 | 3247 | 0 | 7 | 0 |
| normal | 3 | 30 | 0 | 1362 | 264 | 76 |
| other | 30 | 113 | 272 | 49 | 1176 | 77 |
| stroma | 2 | 32 | 4 | 168 | 326 | 550 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8195    0.8442    0.8317       199
      cancer     0.9279    0.8862    0.9066      2645
       empty     0.9201    0.9972    0.9571      3256
      normal     0.8141    0.7850    0.7993      1735
       other     0.6222    0.6849    0.6521      1717
      stroma     0.6782    0.5083    0.5811      1082

    accuracy                         0.8320     10634
   macro avg     0.7970    0.7843    0.7880     10634
weighted avg     0.8302    0.8320    0.8289     10634

```