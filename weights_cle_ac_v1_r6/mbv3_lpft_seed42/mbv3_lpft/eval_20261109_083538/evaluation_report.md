# mbv3_lpft 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_lpft_seed42/mbv3_lpft/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9393 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9913 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9393 |
| Balanced Acc | 0.9368 |
| Macro-Sensitivity | 0.9368 |
| Macro-Specificity | 0.9688 |
| Macro-Precision | 0.9446 |
| AUC-PR (Macro) | 0.9835 |
| Cohen's Kappa | 0.9084 |
| MCC | 0.9098 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 3676.2 (0.27 ms/张) |
| End-to-End FPS | 50.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9266 | 0.8811 | 0.9195 | 0.9771 | 0.9864 | 0.9783 | 208 |
| Malignant_lesions | 0.8750 | 0.9875 | 0.9686 | 0.9194 | 0.9472 | 0.9889 | 0.9749 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9922 | 0.9842 | 0.9791 | 0.9870 | 0.9984 | 0.9971 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=27 TN=341 FN=8
- **Malignant_lesions**: TP=154 FP=5 TN=395 FN=22
- **body_of_stomach_normal**: TP=187 FP=3 TN=381 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 5 | 3 |
| Malignant_lesions | 22 | 154 | 0 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8811    0.9615    0.9195       208
     Malignant_lesions     0.9686    0.8750    0.9194       176
body_of_stomach_normal     0.9842    0.9740    0.9791       192

              accuracy                         0.9392       576
             macro avg     0.9446    0.9368    0.9393       576
          weighted avg     0.9422    0.9392    0.9393       576

```