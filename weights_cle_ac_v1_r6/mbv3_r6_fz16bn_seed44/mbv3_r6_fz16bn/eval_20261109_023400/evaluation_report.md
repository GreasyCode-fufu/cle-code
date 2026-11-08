# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz16bn_seed44/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9604 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9976 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9692 |
| Balanced Acc | 0.9552 |
| Macro-Sensitivity | 0.9552 |
| Macro-Specificity | 0.9835 |
| Macro-Precision | 0.9664 |
| AUC-PR (Macro) | 0.9924 |
| Cohen's Kappa | 0.9514 |
| MCC | 0.9515 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2714.4 (0.37 ms/张) |
| End-to-End FPS | 79.7 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9655 | 0.9598 | 0.9695 | 0.9825 | 0.9976 | 0.9973 | 195 |
| Malignant_lesions | 0.8861 | 0.9885 | 0.9459 | 0.9150 | 0.9745 | 0.9951 | 0.9799 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=8 TN=224 FN=4
- **Malignant_lesions**: TP=70 FP=4 TN=344 FN=9
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 4 | 0 |
| Malignant_lesions | 8 | 70 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9598    0.9795    0.9695       195
     Malignant_lesions     0.9459    0.8861    0.9150        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9696       427
             macro avg     0.9664    0.9552    0.9604       427
          weighted avg     0.9693    0.9696    0.9692       427

```