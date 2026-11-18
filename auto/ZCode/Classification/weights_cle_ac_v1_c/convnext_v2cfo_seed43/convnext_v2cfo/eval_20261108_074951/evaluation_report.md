# convnext_v2cfo 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2cfo_seed43/convnext_v2cfo/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9849 |
| Accuracy | 0.9883 |
| Macro-AUROC (OvR) | 0.9982 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9883 |
| Balanced Acc | 0.9835 |
| Macro-Sensitivity | 0.9835 |
| Macro-Specificity | 0.9938 |
| Macro-Precision | 0.9864 |
| AUC-PR (Macro) | 0.9958 |
| Cohen's Kappa | 0.9814 |
| MCC | 0.9814 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1019.2 (0.98 ms/张) |
| End-to-End FPS | 232.3 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9949 | 0.9871 | 0.9848 | 0.9898 | 0.9957 | 0.9982 | 0.9975 | 195 |
| Malignant_lesions | 0.9620 | 0.9943 | 0.9744 | 0.9682 | 0.9914 | 0.9964 | 0.9900 | 79 |
| body_of_stomach_normal | 0.9935 | 1.0000 | 1.0000 | 0.9967 | 0.9964 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=3 TN=229 FN=1
- **Malignant_lesions**: TP=76 FP=2 TN=346 FN=3
- **body_of_stomach_normal**: TP=152 FP=0 TN=274 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 1 | 0 |
| Malignant_lesions | 3 | 76 | 0 |
| body_of_stomach_normal | 0 | 1 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9848    0.9949    0.9898       195
     Malignant_lesions     0.9744    0.9620    0.9682        79
body_of_stomach_normal     1.0000    0.9935    0.9967       153

              accuracy                         0.9883       427
             macro avg     0.9864    0.9835    0.9849       427
          weighted avg     0.9883    0.9883    0.9883       427

```