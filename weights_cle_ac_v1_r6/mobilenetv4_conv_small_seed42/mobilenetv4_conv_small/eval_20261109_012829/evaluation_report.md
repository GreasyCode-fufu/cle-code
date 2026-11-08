# mobilenetv4_conv_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mobilenetv4_conv_small_seed42/mobilenetv4_conv_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8915 |
| Accuracy | 0.8924 |
| Macro-AUROC (OvR) | 0.9797 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8922 |
| Balanced Acc | 0.8886 |
| Macro-Sensitivity | 0.8886 |
| Macro-Specificity | 0.9452 |
| Macro-Precision | 0.8990 |
| AUC-PR (Macro) | 0.9594 |
| Cohen's Kappa | 0.8376 |
| MCC | 0.8400 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 184.77M |
| FLOPs | 369.55M |
| Params | 2.47M |
| Pure-Inference FPS | 5791.7 (0.17 ms/张) |
| End-to-End FPS | 174.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.8940 | 0.8347 | 0.8874 | 0.9676 | 0.9749 | 0.9627 | 208 |
| Malignant_lesions | 0.8125 | 0.9675 | 0.9167 | 0.8614 | 0.9214 | 0.9736 | 0.9308 | 176 |
| body_of_stomach_normal | 0.9062 | 0.9740 | 0.9457 | 0.9255 | 0.9541 | 0.9906 | 0.9848 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=39 TN=329 FN=11
- **Malignant_lesions**: TP=143 FP=13 TN=387 FN=33
- **body_of_stomach_normal**: TP=174 FP=10 TN=374 FN=18

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 25 | 143 | 8 |
| body_of_stomach_normal | 14 | 4 | 174 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8347    0.9471    0.8874       208
     Malignant_lesions     0.9167    0.8125    0.8614       176
body_of_stomach_normal     0.9457    0.9062    0.9255       192

              accuracy                         0.8924       576
             macro avg     0.8990    0.8886    0.8915       576
          weighted avg     0.8967    0.8924    0.8922       576

```