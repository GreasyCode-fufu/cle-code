# mobilenetv4_conv_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mobilenetv4_conv_small_seed45/mobilenetv4_conv_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9568 |
| Accuracy | 0.9649 |
| Macro-AUROC (OvR) | 0.9934 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9645 |
| Balanced Acc | 0.9518 |
| Macro-Sensitivity | 0.9518 |
| Macro-Specificity | 0.9808 |
| Macro-Precision | 0.9625 |
| AUC-PR (Macro) | 0.9751 |
| Cohen's Kappa | 0.9439 |
| MCC | 0.9441 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 184.77M |
| FLOPs | 369.55M |
| Params | 2.47M |
| Pure-Inference FPS | 5733.1 (0.17 ms/张) |
| End-to-End FPS | 110.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9612 | 0.9545 | 0.9618 | 0.9738 | 0.9923 | 0.9909 | 195 |
| Malignant_lesions | 0.8861 | 0.9885 | 0.9459 | 0.9150 | 0.9745 | 0.9880 | 0.9346 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 1.0000 | 0.9999 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=9 TN=223 FN=6
- **Malignant_lesions**: TP=70 FP=4 TN=344 FN=9
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 4 | 2 |
| Malignant_lesions | 9 | 70 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9545    0.9692    0.9618       195
     Malignant_lesions     0.9459    0.8861    0.9150        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9649       427
             macro avg     0.9625    0.9518    0.9568       427
          weighted avg     0.9646    0.9649    0.9645       427

```