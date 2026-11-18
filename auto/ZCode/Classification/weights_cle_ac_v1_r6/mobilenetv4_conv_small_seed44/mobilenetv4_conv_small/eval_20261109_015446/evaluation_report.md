# mobilenetv4_conv_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mobilenetv4_conv_small_seed44/mobilenetv4_conv_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9546 |
| Accuracy | 0.9625 |
| Macro-AUROC (OvR) | 0.9962 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9624 |
| Balanced Acc | 0.9551 |
| Macro-Sensitivity | 0.9551 |
| Macro-Specificity | 0.9808 |
| Macro-Precision | 0.9541 |
| AUC-PR (Macro) | 0.9888 |
| Cohen's Kappa | 0.9405 |
| MCC | 0.9405 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 184.77M |
| FLOPs | 369.55M |
| Params | 2.47M |
| Pure-Inference FPS | 4860.1 (0.21 ms/张) |
| End-to-End FPS | 102.1 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9538 | 0.9698 | 0.9637 | 0.9588 | 0.9615 | 0.9954 | 0.9946 | 195 |
| Malignant_lesions | 0.9114 | 0.9799 | 0.9114 | 0.9114 | 0.9799 | 0.9932 | 0.9718 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=186 FP=7 TN=225 FN=9
- **Malignant_lesions**: TP=72 FP=7 TN=341 FN=7
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 186 | 7 | 2 |
| Malignant_lesions | 7 | 72 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9637    0.9538    0.9588       195
     Malignant_lesions     0.9114    0.9114    0.9114        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9625       427
             macro avg     0.9541    0.9551    0.9546       427
          weighted avg     0.9624    0.9625    0.9624       427

```