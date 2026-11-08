# mbv3_r6_star2x 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star2x_seed46/mbv3_r6_star2x/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9312 |
| Accuracy | 0.9323 |
| Macro-AUROC (OvR) | 0.9921 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9316 |
| Balanced Acc | 0.9304 |
| Macro-Sensitivity | 0.9304 |
| Macro-Specificity | 0.9657 |
| Macro-Precision | 0.9347 |
| AUC-PR (Macro) | 0.9848 |
| Cohen's Kappa | 0.8980 |
| MCC | 0.8992 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.10M |
| FLOPs | 464.20M |
| Params | 3.39M |
| Pure-Inference FPS | 5846.4 (0.17 ms/张) |
| End-to-End FPS | 293.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9538 | 0.9194 | 0.9260 | 0.9616 | 0.9879 | 0.9796 | 208 |
| Malignant_lesions | 0.8636 | 0.9850 | 0.9620 | 0.9102 | 0.9426 | 0.9893 | 0.9767 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9583 | 0.9227 | 0.9574 | 0.9973 | 0.9990 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=17 TN=351 FN=14
- **Malignant_lesions**: TP=152 FP=6 TN=394 FN=24
- **body_of_stomach_normal**: TP=191 FP=16 TN=368 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 6 | 8 |
| Malignant_lesions | 16 | 152 | 8 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9194    0.9327    0.9260       208
     Malignant_lesions     0.9620    0.8636    0.9102       176
body_of_stomach_normal     0.9227    0.9948    0.9574       192

              accuracy                         0.9323       576
             macro avg     0.9347    0.9304    0.9312       576
          weighted avg     0.9335    0.9323    0.9316       576

```