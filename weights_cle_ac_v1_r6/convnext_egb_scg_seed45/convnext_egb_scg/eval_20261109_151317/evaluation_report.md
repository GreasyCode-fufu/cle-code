# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_egb_scg_seed45/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9800 |
| Accuracy | 0.9836 |
| Macro-AUROC (OvR) | 0.9988 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9835 |
| Balanced Acc | 0.9755 |
| Macro-Sensitivity | 0.9755 |
| Macro-Specificity | 0.9906 |
| Macro-Precision | 0.9850 |
| AUC-PR (Macro) | 0.9960 |
| Cohen's Kappa | 0.9738 |
| MCC | 0.9740 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 266.6 (3.75 ms/张) |
| End-to-End FPS | 141.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9897 | 0.9784 | 0.9747 | 0.9822 | 0.9913 | 0.9986 | 0.9984 | 195 |
| Malignant_lesions | 0.9367 | 0.9971 | 0.9867 | 0.9610 | 0.9858 | 0.9978 | 0.9896 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=5 TN=227 FN=2
- **Malignant_lesions**: TP=74 FP=1 TN=347 FN=5
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 1 | 1 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9747    0.9897    0.9822       195
     Malignant_lesions     0.9867    0.9367    0.9610        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9836       427
             macro avg     0.9850    0.9755    0.9800       427
          weighted avg     0.9837    0.9836    0.9835       427

```