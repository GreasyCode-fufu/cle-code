# convnext_tiny_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_seed43/convnext_tiny_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9426 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9909 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9428 |
| Balanced Acc | 0.9417 |
| Macro-Sensitivity | 0.9417 |
| Macro-Specificity | 0.9710 |
| Macro-Precision | 0.9441 |
| AUC-PR (Macro) | 0.9828 |
| Cohen's Kappa | 0.9138 |
| MCC | 0.9140 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1130.1 (0.88 ms/张) |
| End-to-End FPS | 30.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9484 | 0.9120 | 0.9292 | 0.9694 | 0.9869 | 0.9795 | 208 |
| Malignant_lesions | 0.9091 | 0.9750 | 0.9412 | 0.9249 | 0.9606 | 0.9886 | 0.9741 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9896 | 0.9789 | 0.9738 | 0.9845 | 0.9971 | 0.9948 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=19 TN=349 FN=11
- **Malignant_lesions**: TP=160 FP=10 TN=390 FN=16
- **body_of_stomach_normal**: TP=186 FP=4 TN=380 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 14 | 160 | 2 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9120    0.9471    0.9292       208
     Malignant_lesions     0.9412    0.9091    0.9249       176
body_of_stomach_normal     0.9789    0.9688    0.9738       192

              accuracy                         0.9427       576
             macro avg     0.9441    0.9417    0.9426       576
          weighted avg     0.9432    0.9427    0.9428       576

```