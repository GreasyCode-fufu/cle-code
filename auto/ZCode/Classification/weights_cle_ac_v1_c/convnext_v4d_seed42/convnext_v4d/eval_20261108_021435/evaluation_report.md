# convnext_v4d 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4d_seed42/convnext_v4d/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9662 |
| Accuracy | 0.9742 |
| Macro-AUROC (OvR) | 0.9975 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9738 |
| Balanced Acc | 0.9586 |
| Macro-Sensitivity | 0.9586 |
| Macro-Specificity | 0.9854 |
| Macro-Precision | 0.9753 |
| AUC-PR (Macro) | 0.9919 |
| Cohen's Kappa | 0.9588 |
| MCC | 0.9591 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.59G |
| FLOPs | 7.18G |
| Params | 22.94M |
| Pure-Inference FPS | 536.8 (1.86 ms/张) |
| End-to-End FPS | 192.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9897 | 0.9655 | 0.9602 | 0.9747 | 0.9912 | 0.9968 | 0.9966 | 195 |
| Malignant_lesions | 0.8861 | 0.9943 | 0.9722 | 0.9272 | 0.9746 | 0.9956 | 0.9791 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=8 TN=224 FN=2
- **Malignant_lesions**: TP=70 FP=2 TN=346 FN=9
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 2 | 0 |
| Malignant_lesions | 8 | 70 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9602    0.9897    0.9747       195
     Malignant_lesions     0.9722    0.8861    0.9272        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9742       427
             macro avg     0.9753    0.9586    0.9662       427
          weighted avg     0.9744    0.9742    0.9738       427

```