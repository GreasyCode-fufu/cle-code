# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce16_seed46/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9599 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9945 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9601 |
| Balanced Acc | 0.9601 |
| Macro-Sensitivity | 0.9601 |
| Macro-Specificity | 0.9801 |
| Macro-Precision | 0.9596 |
| AUC-PR (Macro) | 0.9891 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9400 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 5312.4 (0.19 ms/张) |
| End-to-End FPS | 14.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9783 | 0.9612 | 0.9565 | 0.9730 | 0.9917 | 0.9887 | 208 |
| Malignant_lesions | 0.9545 | 0.9750 | 0.9438 | 0.9492 | 0.9799 | 0.9945 | 0.9832 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9870 | 0.9740 | 0.9740 | 0.9870 | 0.9974 | 0.9955 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=8 TN=360 FN=10
- **Malignant_lesions**: TP=168 FP=10 TN=390 FN=8
- **body_of_stomach_normal**: TP=187 FP=5 TN=379 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 9 | 1 |
| Malignant_lesions | 4 | 168 | 4 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9612    0.9519    0.9565       208
     Malignant_lesions     0.9438    0.9545    0.9492       176
body_of_stomach_normal     0.9740    0.9740    0.9740       192

              accuracy                         0.9601       576
             macro avg     0.9596    0.9601    0.9599       576
          weighted avg     0.9601    0.9601    0.9601       576

```