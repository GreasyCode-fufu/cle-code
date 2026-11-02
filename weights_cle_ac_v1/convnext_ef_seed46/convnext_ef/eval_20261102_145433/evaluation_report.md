# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_ef_seed46/convnext_ef/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9581 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9973 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9583 |
| Balanced Acc | 0.9578 |
| Macro-Sensitivity | 0.9578 |
| Macro-Specificity | 0.9790 |
| Macro-Precision | 0.9584 |
| AUC-PR (Macro) | 0.9947 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9374 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 412.9 (2.42 ms/张) |
| End-to-End FPS | 174.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9674 | 0.9429 | 0.9474 | 0.9727 | 0.9961 | 0.9937 | 208 |
| Malignant_lesions | 0.9318 | 0.9775 | 0.9480 | 0.9398 | 0.9702 | 0.9960 | 0.9907 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9922 | 0.9845 | 0.9870 | 0.9948 | 0.9999 | 0.9997 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=12 TN=356 FN=10
- **Malignant_lesions**: TP=164 FP=9 TN=391 FN=12
- **body_of_stomach_normal**: TP=190 FP=3 TN=381 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 9 | 1 |
| Malignant_lesions | 10 | 164 | 2 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9429    0.9519    0.9474       208
     Malignant_lesions     0.9480    0.9318    0.9398       176
body_of_stomach_normal     0.9845    0.9896    0.9870       192

              accuracy                         0.9583       576
             macro avg     0.9584    0.9578    0.9581       576
          weighted avg     0.9583    0.9583    0.9583       576

```