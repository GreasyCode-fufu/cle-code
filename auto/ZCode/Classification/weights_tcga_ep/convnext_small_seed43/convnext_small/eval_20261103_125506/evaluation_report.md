# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_small_seed43/convnext_small/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8129 |
| Accuracy | 0.8432 |
| Macro-AUROC (OvR) | 0.9499 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8359 |
| Balanced Acc | 0.8011 |
| Macro-Sensitivity | 0.8011 |
| Macro-Specificity | 0.9666 |
| Macro-Precision | 0.8371 |
| AUC-PR (Macro) | 0.8574 |
| Cohen's Kappa | 0.7970 |
| MCC | 0.7997 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.42M |
| Pure-Inference FPS | 359.0 (2.79 ms/张) |
| End-to-End FPS | 233.5 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8392 | 0.9976 | 0.8698 | 0.8542 | 0.9969 | 0.9977 | 0.9265 | 199 |
| cancer | 0.9233 | 0.9208 | 0.7941 | 0.8538 | 0.9731 | 0.9714 | 0.9207 | 2645 |
| empty | 0.9963 | 0.9622 | 0.9208 | 0.9571 | 0.9983 | 0.9720 | 0.8701 | 3256 |
| normal | 0.8375 | 0.9631 | 0.8158 | 0.8265 | 0.9681 | 0.9412 | 0.8832 | 1735 |
| other | 0.5533 | 0.9698 | 0.7793 | 0.6471 | 0.9185 | 0.8677 | 0.7352 | 1717 |
| stroma | 0.6571 | 0.9861 | 0.8424 | 0.7383 | 0.9621 | 0.9497 | 0.8084 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=167 FP=25 TN=10410 FN=32
- **cancer**: TP=2442 FP=633 TN=7356 FN=203
- **empty**: TP=3244 FP=279 TN=7099 FN=12
- **normal**: TP=1453 FP=328 TN=8571 FN=282
- **other**: TP=950 FP=269 TN=8648 FN=767
- **stroma**: TP=711 FP=133 TN=9419 FN=371

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 167 | 5 | 2 | 9 | 15 | 1 |
| cancer | 2 | 2442 | 0 | 135 | 62 | 4 |
| empty | 0 | 0 | 3244 | 0 | 11 | 1 |
| normal | 1 | 197 | 1 | 1453 | 58 | 25 |
| other | 18 | 306 | 276 | 65 | 950 | 102 |
| stroma | 4 | 125 | 0 | 119 | 123 | 711 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8698    0.8392    0.8542       199
      cancer     0.7941    0.9233    0.8538      2645
       empty     0.9208    0.9963    0.9571      3256
      normal     0.8158    0.8375    0.8265      1735
       other     0.7793    0.5533    0.6471      1717
      stroma     0.8424    0.6571    0.7383      1082

    accuracy                         0.8432     10634
   macro avg     0.8371    0.8011    0.8129     10634
weighted avg     0.8404    0.8432    0.8359     10634

```