# resnext50_32x4d 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/resnext50_32x4d_seed43/resnext50_32x4d/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9614 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9971 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9696 |
| Balanced Acc | 0.9627 |
| Macro-Sensitivity | 0.9627 |
| Macro-Specificity | 0.9849 |
| Macro-Precision | 0.9600 |
| AUC-PR (Macro) | 0.9910 |
| Cohen's Kappa | 0.9517 |
| MCC | 0.9517 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.29G |
| FLOPs | 8.57G |
| Params | 22.99M |
| Pure-Inference FPS | 950.6 (1.05 ms/张) |
| End-to-End FPS | 209.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9641 | 0.9784 | 0.9741 | 0.9691 | 0.9701 | 0.9968 | 0.9963 | 195 |
| Malignant_lesions | 0.9241 | 0.9799 | 0.9125 | 0.9182 | 0.9827 | 0.9943 | 0.9768 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=5 TN=227 FN=7
- **Malignant_lesions**: TP=73 FP=7 TN=341 FN=6
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 7 | 0 |
| Malignant_lesions | 5 | 73 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9741    0.9641    0.9691       195
     Malignant_lesions     0.9125    0.9241    0.9182        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9696       427
             macro avg     0.9600    0.9627    0.9614       427
          weighted avg     0.9697    0.9696    0.9696       427

```