# mbv3_lpft 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_lpft_seed44/mbv3_lpft/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9461 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9911 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9462 |
| Balanced Acc | 0.9453 |
| Macro-Sensitivity | 0.9453 |
| Macro-Specificity | 0.9728 |
| Macro-Precision | 0.9472 |
| AUC-PR (Macro) | 0.9836 |
| Cohen's Kappa | 0.9190 |
| MCC | 0.9192 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 3906.9 (0.26 ms/张) |
| End-to-End FPS | 134.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9538 | 0.9206 | 0.9336 | 0.9696 | 0.9884 | 0.9800 | 208 |
| Malignant_lesions | 0.9148 | 0.9775 | 0.9471 | 0.9306 | 0.9631 | 0.9860 | 0.9729 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9870 | 0.9740 | 0.9740 | 0.9870 | 0.9990 | 0.9980 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=17 TN=351 FN=11
- **Malignant_lesions**: TP=161 FP=9 TN=391 FN=15
- **body_of_stomach_normal**: TP=187 FP=5 TN=379 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 8 | 3 |
| Malignant_lesions | 13 | 161 | 2 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9206    0.9471    0.9336       208
     Malignant_lesions     0.9471    0.9148    0.9306       176
body_of_stomach_normal     0.9740    0.9740    0.9740       192

              accuracy                         0.9462       576
             macro avg     0.9472    0.9453    0.9461       576
          weighted avg     0.9465    0.9462    0.9462       576

```