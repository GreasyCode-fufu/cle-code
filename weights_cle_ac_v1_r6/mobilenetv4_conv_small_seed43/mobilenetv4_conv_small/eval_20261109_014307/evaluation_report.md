# mobilenetv4_conv_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mobilenetv4_conv_small_seed43/mobilenetv4_conv_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9675 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9979 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9719 |
| Balanced Acc | 0.9694 |
| Macro-Sensitivity | 0.9694 |
| Macro-Specificity | 0.9858 |
| Macro-Precision | 0.9658 |
| AUC-PR (Macro) | 0.9936 |
| Cohen's Kappa | 0.9554 |
| MCC | 0.9555 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 184.77M |
| FLOPs | 369.55M |
| Params | 2.47M |
| Pure-Inference FPS | 5304.6 (0.19 ms/张) |
| End-to-End FPS | 135.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9828 | 0.9791 | 0.9689 | 0.9661 | 0.9973 | 0.9970 | 195 |
| Malignant_lesions | 0.9494 | 0.9856 | 0.9375 | 0.9434 | 0.9885 | 0.9965 | 0.9840 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9891 | 0.9808 | 0.9903 | 1.0000 | 1.0000 | 0.9999 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=4 TN=228 FN=8
- **Malignant_lesions**: TP=75 FP=5 TN=343 FN=4
- **body_of_stomach_normal**: TP=153 FP=3 TN=271 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 5 | 3 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9791    0.9590    0.9689       195
     Malignant_lesions     0.9375    0.9494    0.9434        79
body_of_stomach_normal     0.9808    1.0000    0.9903       153

              accuracy                         0.9719       427
             macro avg     0.9658    0.9694    0.9675       427
          weighted avg     0.9720    0.9719    0.9719       427

```