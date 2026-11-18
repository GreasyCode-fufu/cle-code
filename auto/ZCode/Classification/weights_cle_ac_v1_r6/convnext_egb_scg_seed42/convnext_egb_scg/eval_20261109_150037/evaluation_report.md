# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_egb_scg_seed42/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9561 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9959 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9567 |
| Balanced Acc | 0.9579 |
| Macro-Sensitivity | 0.9579 |
| Macro-Specificity | 0.9789 |
| Macro-Precision | 0.9561 |
| AUC-PR (Macro) | 0.9924 |
| Cohen's Kappa | 0.9349 |
| MCC | 0.9358 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 251.1 (3.98 ms/张) |
| End-to-End FPS | 137.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9183 | 0.9973 | 0.9948 | 0.9550 | 0.9557 | 0.9959 | 0.9937 | 208 |
| Malignant_lesions | 0.9659 | 0.9550 | 0.9043 | 0.9341 | 0.9845 | 0.9929 | 0.9859 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9844 | 0.9694 | 0.9794 | 0.9947 | 0.9988 | 0.9975 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=1 TN=367 FN=17
- **Malignant_lesions**: TP=170 FP=18 TN=382 FN=6
- **body_of_stomach_normal**: TP=190 FP=6 TN=378 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 16 | 1 |
| Malignant_lesions | 1 | 170 | 5 |
| body_of_stomach_normal | 0 | 2 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9948    0.9183    0.9550       208
     Malignant_lesions     0.9043    0.9659    0.9341       176
body_of_stomach_normal     0.9694    0.9896    0.9794       192

              accuracy                         0.9566       576
             macro avg     0.9561    0.9579    0.9561       576
          weighted avg     0.9587    0.9566    0.9567       576

```