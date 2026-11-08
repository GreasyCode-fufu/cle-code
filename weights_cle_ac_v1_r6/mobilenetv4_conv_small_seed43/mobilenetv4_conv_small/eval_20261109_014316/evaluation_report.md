# mobilenetv4_conv_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mobilenetv4_conv_small_seed43/mobilenetv4_conv_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9093 |
| Accuracy | 0.9097 |
| Macro-AUROC (OvR) | 0.9845 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9097 |
| Balanced Acc | 0.9087 |
| Macro-Sensitivity | 0.9087 |
| Macro-Specificity | 0.9545 |
| Macro-Precision | 0.9101 |
| AUC-PR (Macro) | 0.9711 |
| Cohen's Kappa | 0.8642 |
| MCC | 0.8643 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 184.77M |
| FLOPs | 369.55M |
| Params | 2.47M |
| Pure-Inference FPS | 8368.6 (0.12 ms/张) |
| End-to-End FPS | 115.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8990 | 0.9293 | 0.8779 | 0.8884 | 0.9421 | 0.9790 | 0.9642 | 208 |
| Malignant_lesions | 0.8636 | 0.9525 | 0.8889 | 0.8761 | 0.9407 | 0.9775 | 0.9548 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9818 | 0.9635 | 0.9635 | 0.9818 | 0.9970 | 0.9943 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=26 TN=342 FN=21
- **Malignant_lesions**: TP=152 FP=19 TN=381 FN=24
- **body_of_stomach_normal**: TP=185 FP=7 TN=377 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 18 | 3 |
| Malignant_lesions | 20 | 152 | 4 |
| body_of_stomach_normal | 6 | 1 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8779    0.8990    0.8884       208
     Malignant_lesions     0.8889    0.8636    0.8761       176
body_of_stomach_normal     0.9635    0.9635    0.9635       192

              accuracy                         0.9097       576
             macro avg     0.9101    0.9087    0.9093       576
          weighted avg     0.9098    0.9097    0.9097       576

```