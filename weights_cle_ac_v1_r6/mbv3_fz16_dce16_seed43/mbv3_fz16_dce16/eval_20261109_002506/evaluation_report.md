# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce16_seed43/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9585 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9583 |
| Balanced Acc | 0.9585 |
| Macro-Sensitivity | 0.9585 |
| Macro-Specificity | 0.9790 |
| Macro-Precision | 0.9585 |
| AUC-PR (Macro) | 0.9923 |
| Cohen's Kappa | 0.9374 |
| MCC | 0.9374 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 5540.1 (0.18 ms/张) |
| End-to-End FPS | 14.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9701 | 0.9471 | 0.9471 | 0.9701 | 0.9936 | 0.9912 | 208 |
| Malignant_lesions | 0.9545 | 0.9800 | 0.9545 | 0.9545 | 0.9800 | 0.9956 | 0.9877 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9870 | 0.9740 | 0.9740 | 0.9870 | 0.9990 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=11 TN=357 FN=11
- **Malignant_lesions**: TP=168 FP=8 TN=392 FN=8
- **body_of_stomach_normal**: TP=187 FP=5 TN=379 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 8 | 3 |
| Malignant_lesions | 6 | 168 | 2 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9471    0.9471    0.9471       208
     Malignant_lesions     0.9545    0.9545    0.9545       176
body_of_stomach_normal     0.9740    0.9740    0.9740       192

              accuracy                         0.9583       576
             macro avg     0.9585    0.9585    0.9585       576
          weighted avg     0.9583    0.9583    0.9583       576

```