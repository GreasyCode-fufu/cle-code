# mbv3_lpft 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_lpft_seed43/mbv3_lpft/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9367 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9912 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9372 |
| Balanced Acc | 0.9342 |
| Macro-Sensitivity | 0.9342 |
| Macro-Specificity | 0.9679 |
| Macro-Precision | 0.9431 |
| AUC-PR (Macro) | 0.9837 |
| Cohen's Kappa | 0.9057 |
| MCC | 0.9076 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 3110.3 (0.32 ms/张) |
| End-to-End FPS | 63.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9293 | 0.8865 | 0.9291 | 0.9856 | 0.9885 | 0.9816 | 208 |
| Malignant_lesions | 0.8580 | 0.9900 | 0.9742 | 0.9124 | 0.9406 | 0.9870 | 0.9729 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9844 | 0.9688 | 0.9688 | 0.9844 | 0.9981 | 0.9965 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=26 TN=342 FN=5
- **Malignant_lesions**: TP=151 FP=4 TN=396 FN=25
- **body_of_stomach_normal**: TP=186 FP=6 TN=378 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 4 | 1 |
| Malignant_lesions | 20 | 151 | 5 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8865    0.9760    0.9291       208
     Malignant_lesions     0.9742    0.8580    0.9124       176
body_of_stomach_normal     0.9688    0.9688    0.9688       192

              accuracy                         0.9375       576
             macro avg     0.9431    0.9342    0.9367       576
          weighted avg     0.9407    0.9375    0.9372       576

```