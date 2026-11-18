# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_small_seed42/convnext_small/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8068 |
| Accuracy | 0.8462 |
| Macro-AUROC (OvR) | 0.9639 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8400 |
| Balanced Acc | 0.8011 |
| Macro-Sensitivity | 0.8011 |
| Macro-Specificity | 0.9677 |
| Macro-Precision | 0.8188 |
| AUC-PR (Macro) | 0.8727 |
| Cohen's Kappa | 0.8013 |
| MCC | 0.8025 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.42M |
| Pure-Inference FPS | 414.9 (2.41 ms/张) |
| End-to-End FPS | 255.7 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8392 | 0.9964 | 0.8146 | 0.8267 | 0.9969 | 0.9971 | 0.9116 | 199 |
| cancer | 0.9096 | 0.9443 | 0.8439 | 0.8755 | 0.9693 | 0.9799 | 0.9366 | 2645 |
| empty | 0.9963 | 0.9616 | 0.9198 | 0.9565 | 0.9983 | 0.9811 | 0.9472 | 3256 |
| normal | 0.8617 | 0.9694 | 0.8461 | 0.8538 | 0.9729 | 0.9763 | 0.9300 | 1735 |
| other | 0.6104 | 0.9548 | 0.7223 | 0.6616 | 0.9271 | 0.8936 | 0.7384 | 1717 |
| stroma | 0.5896 | 0.9796 | 0.7659 | 0.6663 | 0.9547 | 0.9556 | 0.7724 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=167 FP=38 TN=10397 FN=32
- **cancer**: TP=2406 FP=445 TN=7544 FN=239
- **empty**: TP=3244 FP=283 TN=7095 FN=12
- **normal**: TP=1495 FP=272 TN=8627 FN=240
- **other**: TP=1048 FP=403 TN=8514 FN=669
- **stroma**: TP=638 FP=195 TN=9357 FN=444

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 167 | 3 | 2 | 8 | 18 | 1 |
| cancer | 3 | 2406 | 1 | 83 | 113 | 39 |
| empty | 0 | 1 | 3244 | 0 | 11 | 0 |
| normal | 2 | 153 | 1 | 1495 | 36 | 48 |
| other | 31 | 191 | 279 | 61 | 1048 | 107 |
| stroma | 2 | 97 | 0 | 120 | 225 | 638 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8146    0.8392    0.8267       199
      cancer     0.8439    0.9096    0.8755      2645
       empty     0.9198    0.9963    0.9565      3256
      normal     0.8461    0.8617    0.8538      1735
       other     0.7223    0.6104    0.6616      1717
      stroma     0.7659    0.5896    0.6663      1082

    accuracy                         0.8462     10634
   macro avg     0.8188    0.8011    0.8068     10634
weighted avg     0.8394    0.8462    0.8400     10634

```