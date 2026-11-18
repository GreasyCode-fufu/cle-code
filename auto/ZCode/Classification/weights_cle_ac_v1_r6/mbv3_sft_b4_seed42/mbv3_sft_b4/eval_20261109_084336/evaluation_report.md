# mbv3_sft_b4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b4_seed42/mbv3_sft_b4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9206 |
| Accuracy | 0.9201 |
| Macro-AUROC (OvR) | 0.9866 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9206 |
| Balanced Acc | 0.9167 |
| Macro-Sensitivity | 0.9167 |
| Macro-Specificity | 0.9588 |
| Macro-Precision | 0.9307 |
| AUC-PR (Macro) | 0.9760 |
| Cohen's Kappa | 0.8795 |
| MCC | 0.8826 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2172.8 (0.46 ms/张) |
| End-to-End FPS | 29.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.8967 | 0.8410 | 0.8993 | 0.9792 | 0.9804 | 0.9691 | 208 |
| Malignant_lesions | 0.8409 | 0.9875 | 0.9673 | 0.8997 | 0.9338 | 0.9826 | 0.9650 | 176 |
| body_of_stomach_normal | 0.9427 | 0.9922 | 0.9837 | 0.9628 | 0.9719 | 0.9968 | 0.9939 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=38 TN=330 FN=7
- **Malignant_lesions**: TP=148 FP=5 TN=395 FN=28
- **body_of_stomach_normal**: TP=181 FP=3 TN=381 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 27 | 148 | 1 |
| body_of_stomach_normal | 11 | 0 | 181 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8410    0.9663    0.8993       208
     Malignant_lesions     0.9673    0.8409    0.8997       176
body_of_stomach_normal     0.9837    0.9427    0.9628       192

              accuracy                         0.9201       576
             macro avg     0.9307    0.9167    0.9206       576
          weighted avg     0.9272    0.9201    0.9206       576

```