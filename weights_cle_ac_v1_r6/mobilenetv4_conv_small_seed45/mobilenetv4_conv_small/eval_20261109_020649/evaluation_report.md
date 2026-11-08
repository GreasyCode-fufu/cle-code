# mobilenetv4_conv_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mobilenetv4_conv_small_seed45/mobilenetv4_conv_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8789 |
| Accuracy | 0.8802 |
| Macro-AUROC (OvR) | 0.9766 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8796 |
| Balanced Acc | 0.8744 |
| Macro-Sensitivity | 0.8744 |
| Macro-Specificity | 0.9383 |
| Macro-Precision | 0.8973 |
| AUC-PR (Macro) | 0.9590 |
| Cohen's Kappa | 0.8188 |
| MCC | 0.8253 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 184.77M |
| FLOPs | 369.55M |
| Params | 2.47M |
| Pure-Inference FPS | 7123.7 (0.14 ms/张) |
| End-to-End FPS | 120.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.8560 | 0.7888 | 0.8627 | 0.9692 | 0.9658 | 0.9454 | 208 |
| Malignant_lesions | 0.7443 | 0.9850 | 0.9562 | 0.8371 | 0.8975 | 0.9719 | 0.9467 | 176 |
| body_of_stomach_normal | 0.9271 | 0.9740 | 0.9468 | 0.9368 | 0.9639 | 0.9921 | 0.9848 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=53 TN=315 FN=10
- **Malignant_lesions**: TP=131 FP=6 TN=394 FN=45
- **body_of_stomach_normal**: TP=178 FP=10 TN=374 FN=14

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 5 | 5 |
| Malignant_lesions | 40 | 131 | 5 |
| body_of_stomach_normal | 13 | 1 | 178 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.7888    0.9519    0.8627       208
     Malignant_lesions     0.9562    0.7443    0.8371       176
body_of_stomach_normal     0.9468    0.9271    0.9368       192

              accuracy                         0.8802       576
             macro avg     0.8973    0.8744    0.8789       576
          weighted avg     0.8926    0.8802    0.8796       576

```