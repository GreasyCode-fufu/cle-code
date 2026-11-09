# mbv3_crt 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_crt_seed45/mbv3_crt/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9642 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9969 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9719 |
| Balanced Acc | 0.9644 |
| Macro-Sensitivity | 0.9644 |
| Macro-Specificity | 0.9859 |
| Macro-Precision | 0.9639 |
| AUC-PR (Macro) | 0.9903 |
| Cohen's Kappa | 0.9553 |
| MCC | 0.9553 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 1426.6 (0.70 ms/张) |
| End-to-End FPS | 7.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9784 | 0.9742 | 0.9717 | 0.9742 | 0.9970 | 0.9965 | 195 |
| Malignant_lesions | 0.9241 | 0.9828 | 0.9241 | 0.9241 | 0.9828 | 0.9936 | 0.9745 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=5 TN=227 FN=6
- **Malignant_lesions**: TP=73 FP=6 TN=342 FN=6
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 6 | 0 |
| Malignant_lesions | 5 | 73 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9742    0.9692    0.9717       195
     Malignant_lesions     0.9241    0.9241    0.9241        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9719       427
             macro avg     0.9639    0.9644    0.9642       427
          weighted avg     0.9719    0.9719    0.9719       427

```