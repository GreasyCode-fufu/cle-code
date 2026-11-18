# convnext_v4d 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4d_seed42/convnext_v4d/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9154 |
| Accuracy | 0.9167 |
| Macro-AUROC (OvR) | 0.9935 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9161 |
| Balanced Acc | 0.9116 |
| Macro-Sensitivity | 0.9116 |
| Macro-Specificity | 0.9569 |
| Macro-Precision | 0.9292 |
| AUC-PR (Macro) | 0.9876 |
| Cohen's Kappa | 0.8741 |
| MCC | 0.8788 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.59G |
| FLOPs | 7.18G |
| Params | 22.94M |
| Pure-Inference FPS | 628.3 (1.59 ms/张) |
| End-to-End FPS | 204.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.8886 | 0.8313 | 0.8958 | 0.9820 | 0.9895 | 0.9821 | 208 |
| Malignant_lesions | 0.7898 | 0.9900 | 0.9720 | 0.8715 | 0.9145 | 0.9914 | 0.9815 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9922 | 0.9842 | 0.9791 | 0.9870 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=41 TN=327 FN=6
- **Malignant_lesions**: TP=139 FP=4 TN=396 FN=37
- **body_of_stomach_normal**: TP=187 FP=3 TN=381 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 36 | 139 | 1 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8313    0.9712    0.8958       208
     Malignant_lesions     0.9720    0.7898    0.8715       176
body_of_stomach_normal     0.9842    0.9740    0.9791       192

              accuracy                         0.9167       576
             macro avg     0.9292    0.9116    0.9154       576
          weighted avg     0.9253    0.9167    0.9161       576

```