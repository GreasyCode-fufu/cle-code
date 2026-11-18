# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce16_seed44/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9568 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9940 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9566 |
| Balanced Acc | 0.9577 |
| Macro-Sensitivity | 0.9577 |
| Macro-Specificity | 0.9784 |
| Macro-Precision | 0.9564 |
| AUC-PR (Macro) | 0.9883 |
| Cohen's Kappa | 0.9348 |
| MCC | 0.9350 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 5332.7 (0.19 ms/张) |
| End-to-End FPS | 14.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9755 | 0.9557 | 0.9440 | 0.9625 | 0.9919 | 0.9868 | 208 |
| Malignant_lesions | 0.9716 | 0.9675 | 0.9293 | 0.9500 | 0.9872 | 0.9939 | 0.9826 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9922 | 0.9841 | 0.9764 | 0.9845 | 0.9961 | 0.9953 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=9 TN=359 FN=14
- **Malignant_lesions**: TP=171 FP=13 TN=387 FN=5
- **body_of_stomach_normal**: TP=186 FP=3 TN=381 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 12 | 2 |
| Malignant_lesions | 4 | 171 | 1 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9557    0.9327    0.9440       208
     Malignant_lesions     0.9293    0.9716    0.9500       176
body_of_stomach_normal     0.9841    0.9688    0.9764       192

              accuracy                         0.9566       576
             macro avg     0.9564    0.9577    0.9568       576
          weighted avg     0.9571    0.9566    0.9566       576

```