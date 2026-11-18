# convnext_v5se 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5se_seed43/convnext_v5se/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9668 |
| Accuracy | 0.9670 |
| Macro-AUROC (OvR) | 0.9958 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9669 |
| Balanced Acc | 0.9674 |
| Macro-Sensitivity | 0.9674 |
| Macro-Specificity | 0.9836 |
| Macro-Precision | 0.9664 |
| AUC-PR (Macro) | 0.9918 |
| Cohen's Kappa | 0.9504 |
| MCC | 0.9506 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.47M |
| Pure-Inference FPS | 484.3 (2.06 ms/张) |
| End-to-End FPS | 187.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9864 | 0.9752 | 0.9610 | 0.9706 | 0.9942 | 0.9905 | 208 |
| Malignant_lesions | 0.9602 | 0.9775 | 0.9494 | 0.9548 | 0.9824 | 0.9939 | 0.9859 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9870 | 0.9745 | 0.9845 | 0.9974 | 0.9994 | 0.9988 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=5 TN=363 FN=11
- **Malignant_lesions**: TP=169 FP=9 TN=391 FN=7
- **body_of_stomach_normal**: TP=191 FP=5 TN=379 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 4 | 169 | 3 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9752    0.9471    0.9610       208
     Malignant_lesions     0.9494    0.9602    0.9548       176
body_of_stomach_normal     0.9745    0.9948    0.9845       192

              accuracy                         0.9670       576
             macro avg     0.9664    0.9674    0.9668       576
          weighted avg     0.9671    0.9670    0.9669       576

```