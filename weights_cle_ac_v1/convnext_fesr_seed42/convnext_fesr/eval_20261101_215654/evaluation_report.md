# convnext_fesr 评估报告

- 权重: `weights_cle_ac_v1/convnext_fesr_seed42/convnext_fesr/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9427 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9956 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9430 |
| Balanced Acc | 0.9432 |
| Macro-Sensitivity | 0.9432 |
| Macro-Specificity | 0.9715 |
| Macro-Precision | 0.9437 |
| AUC-PR (Macro) | 0.9917 |
| Cohen's Kappa | 0.9139 |
| MCC | 0.9146 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1337.0 (0.75 ms/张) |
| End-to-End FPS | 224.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9647 | 0.9378 | 0.9400 | 0.9673 | 0.9931 | 0.9892 | 208 |
| Malignant_lesions | 0.9602 | 0.9525 | 0.8989 | 0.9286 | 0.9820 | 0.9947 | 0.9879 | 176 |
| body_of_stomach_normal | 0.9271 | 0.9974 | 0.9944 | 0.9596 | 0.9647 | 0.9989 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=13 TN=355 FN=12
- **Malignant_lesions**: TP=169 FP=19 TN=381 FN=7
- **body_of_stomach_normal**: TP=178 FP=1 TN=383 FN=14

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 11 | 1 |
| Malignant_lesions | 7 | 169 | 0 |
| body_of_stomach_normal | 6 | 8 | 178 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9378    0.9423    0.9400       208
     Malignant_lesions     0.8989    0.9602    0.9286       176
body_of_stomach_normal     0.9944    0.9271    0.9596       192

              accuracy                         0.9427       576
             macro avg     0.9437    0.9432    0.9427       576
          weighted avg     0.9448    0.9427    0.9430       576

```