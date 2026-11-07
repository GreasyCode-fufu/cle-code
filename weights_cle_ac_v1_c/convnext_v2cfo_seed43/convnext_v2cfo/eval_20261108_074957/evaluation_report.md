# convnext_v2cfo 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2cfo_seed43/convnext_v2cfo/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9200 |
| Accuracy | 0.9201 |
| Macro-AUROC (OvR) | 0.9938 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9203 |
| Balanced Acc | 0.9181 |
| Macro-Sensitivity | 0.9181 |
| Macro-Specificity | 0.9595 |
| Macro-Precision | 0.9264 |
| AUC-PR (Macro) | 0.9886 |
| Cohen's Kappa | 0.8797 |
| MCC | 0.8820 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 957.2 (1.04 ms/张) |
| End-to-End FPS | 236.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9185 | 0.8712 | 0.9206 | 0.9854 | 0.9918 | 0.9871 | 208 |
| Malignant_lesions | 0.9034 | 0.9625 | 0.9138 | 0.9086 | 0.9577 | 0.9938 | 0.9855 | 176 |
| body_of_stomach_normal | 0.8750 | 0.9974 | 0.9941 | 0.9307 | 0.9410 | 0.9958 | 0.9931 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=30 TN=338 FN=5
- **Malignant_lesions**: TP=159 FP=15 TN=385 FN=17
- **body_of_stomach_normal**: TP=168 FP=1 TN=383 FN=24

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 5 | 0 |
| Malignant_lesions | 16 | 159 | 1 |
| body_of_stomach_normal | 14 | 10 | 168 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8712    0.9760    0.9206       208
     Malignant_lesions     0.9138    0.9034    0.9086       176
body_of_stomach_normal     0.9941    0.8750    0.9307       192

              accuracy                         0.9201       576
             macro avg     0.9264    0.9181    0.9200       576
          weighted avg     0.9252    0.9201    0.9203       576

```