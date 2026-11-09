# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/gmnet_s3_seed42/gmnet_s3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9646 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9717 |
| Balanced Acc | 0.9619 |
| Macro-Sensitivity | 0.9619 |
| Macro-Specificity | 0.9853 |
| Macro-Precision | 0.9675 |
| AUC-PR (Macro) | 0.9894 |
| Cohen's Kappa | 0.9552 |
| MCC | 0.9553 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 1074.2 (0.93 ms/张) |
| End-to-End FPS | 225.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9784 | 0.9744 | 0.9744 | 0.9784 | 0.9959 | 0.9960 | 195 |
| Malignant_lesions | 0.9114 | 0.9885 | 0.9474 | 0.9290 | 0.9801 | 0.9945 | 0.9722 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9891 | 0.9808 | 0.9903 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=5 TN=227 FN=5
- **Malignant_lesions**: TP=72 FP=4 TN=344 FN=7
- **body_of_stomach_normal**: TP=153 FP=3 TN=271 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 4 | 1 |
| Malignant_lesions | 5 | 72 | 2 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9744    0.9744    0.9744       195
     Malignant_lesions     0.9474    0.9114    0.9290        79
body_of_stomach_normal     0.9808    1.0000    0.9903       153

              accuracy                         0.9719       427
             macro avg     0.9675    0.9619    0.9646       427
          weighted avg     0.9717    0.9719    0.9717       427

```