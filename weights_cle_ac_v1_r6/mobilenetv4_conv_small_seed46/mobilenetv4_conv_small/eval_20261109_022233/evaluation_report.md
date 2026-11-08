# mobilenetv4_conv_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mobilenetv4_conv_small_seed46/mobilenetv4_conv_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9121 |
| Accuracy | 0.9149 |
| Macro-AUROC (OvR) | 0.9902 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9134 |
| Balanced Acc | 0.9105 |
| Macro-Sensitivity | 0.9105 |
| Macro-Specificity | 0.9566 |
| Macro-Precision | 0.9206 |
| AUC-PR (Macro) | 0.9818 |
| Cohen's Kappa | 0.8716 |
| MCC | 0.8744 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 184.77M |
| FLOPs | 369.55M |
| Params | 2.47M |
| Pure-Inference FPS | 7556.9 (0.13 ms/张) |
| End-to-End FPS | 32.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9239 | 0.8767 | 0.9149 | 0.9742 | 0.9865 | 0.9808 | 208 |
| Malignant_lesions | 0.7955 | 0.9850 | 0.9589 | 0.8696 | 0.9163 | 0.9882 | 0.9723 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9609 | 0.9261 | 0.9519 | 0.9893 | 0.9961 | 0.9923 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=28 TN=340 FN=9
- **Malignant_lesions**: TP=140 FP=6 TN=394 FN=36
- **body_of_stomach_normal**: TP=188 FP=15 TN=369 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 6 | 3 |
| Malignant_lesions | 24 | 140 | 12 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8767    0.9567    0.9149       208
     Malignant_lesions     0.9589    0.7955    0.8696       176
body_of_stomach_normal     0.9261    0.9792    0.9519       192

              accuracy                         0.9149       576
             macro avg     0.9206    0.9105    0.9121       576
          weighted avg     0.9183    0.9149    0.9134       576

```