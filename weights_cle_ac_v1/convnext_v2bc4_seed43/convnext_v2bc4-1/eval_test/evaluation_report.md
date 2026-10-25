# convnext_v2bc4 评估报告

- 权重: `weights_cle_ac_v1/convnext_v2bc4_seed43/convnext_v2bc4-1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9580 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9964 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9582 |
| Balanced Acc | 0.9579 |
| Macro-Sensitivity | 0.9579 |
| Macro-Specificity | 0.9791 |
| Macro-Precision | 0.9582 |
| AUC-PR (Macro) | 0.9933 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9374 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.42M |
| Pure-Inference FPS | 1626.4 (0.61 ms/张) |
| End-to-End FPS | 229.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9701 | 0.9471 | 0.9471 | 0.9701 | 0.9950 | 0.9916 | 208 |
| Malignant_lesions | 0.9318 | 0.9775 | 0.9480 | 0.9398 | 0.9702 | 0.9946 | 0.9888 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9896 | 0.9795 | 0.9871 | 0.9974 | 0.9997 | 0.9993 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=11 TN=357 FN=11
- **Malignant_lesions**: TP=164 FP=9 TN=391 FN=12
- **body_of_stomach_normal**: TP=191 FP=4 TN=380 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 10 | 164 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9471    0.9471    0.9471       208
     Malignant_lesions     0.9480    0.9318    0.9398       176
body_of_stomach_normal     0.9795    0.9948    0.9871       192

              accuracy                         0.9583       576
             macro avg     0.9582    0.9579    0.9580       576
          weighted avg     0.9582    0.9583    0.9582       576

```