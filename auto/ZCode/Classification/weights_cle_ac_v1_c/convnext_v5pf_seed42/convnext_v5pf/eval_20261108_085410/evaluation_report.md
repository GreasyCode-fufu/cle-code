# convnext_v5pf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5pf_seed42/convnext_v5pf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9490 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9946 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9496 |
| Balanced Acc | 0.9488 |
| Macro-Sensitivity | 0.9488 |
| Macro-Specificity | 0.9748 |
| Macro-Precision | 0.9493 |
| AUC-PR (Macro) | 0.9897 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9243 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 635.5 (1.57 ms/张) |
| End-to-End FPS | 209.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9647 | 0.9381 | 0.9426 | 0.9699 | 0.9930 | 0.9891 | 208 |
| Malignant_lesions | 0.9148 | 0.9675 | 0.9253 | 0.9200 | 0.9627 | 0.9918 | 0.9820 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9922 | 0.9844 | 0.9844 | 0.9922 | 0.9991 | 0.9980 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=13 TN=355 FN=11
- **Malignant_lesions**: TP=161 FP=13 TN=387 FN=15
- **body_of_stomach_normal**: TP=189 FP=3 TN=381 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 13 | 161 | 2 |
| body_of_stomach_normal | 0 | 3 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9381    0.9471    0.9426       208
     Malignant_lesions     0.9253    0.9148    0.9200       176
body_of_stomach_normal     0.9844    0.9844    0.9844       192

              accuracy                         0.9497       576
             macro avg     0.9493    0.9488    0.9490       576
          weighted avg     0.9496    0.9497    0.9496       576

```