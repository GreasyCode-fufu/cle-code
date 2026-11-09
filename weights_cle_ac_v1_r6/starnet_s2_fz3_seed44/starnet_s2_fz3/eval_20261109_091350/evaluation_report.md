# starnet_s2_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_seed44/starnet_s2_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9626 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9715 |
| Balanced Acc | 0.9569 |
| Macro-Sensitivity | 0.9569 |
| Macro-Specificity | 0.9851 |
| Macro-Precision | 0.9693 |
| AUC-PR (Macro) | 0.9863 |
| Cohen's Kappa | 0.9551 |
| MCC | 0.9553 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.29M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 2971.8 (0.34 ms/张) |
| End-to-End FPS | 98.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9784 | 0.9746 | 0.9796 | 0.9870 | 0.9950 | 0.9938 | 195 |
| Malignant_lesions | 0.8861 | 0.9914 | 0.9589 | 0.9211 | 0.9746 | 0.9897 | 0.9651 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9854 | 0.9745 | 0.9871 | 1.0000 | 1.0000 | 0.9999 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=5 TN=227 FN=3
- **Malignant_lesions**: TP=70 FP=3 TN=345 FN=9
- **body_of_stomach_normal**: TP=153 FP=4 TN=270 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 3 | 0 |
| Malignant_lesions | 5 | 70 | 4 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9746    0.9846    0.9796       195
     Malignant_lesions     0.9589    0.8861    0.9211        79
body_of_stomach_normal     0.9745    1.0000    0.9871       153

              accuracy                         0.9719       427
             macro avg     0.9693    0.9569    0.9626       427
          weighted avg     0.9717    0.9719    0.9715       427

```