# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce4s_seed44/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9603 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9943 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9601 |
| Balanced Acc | 0.9609 |
| Macro-Sensitivity | 0.9609 |
| Macro-Specificity | 0.9801 |
| Macro-Precision | 0.9599 |
| AUC-PR (Macro) | 0.9892 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9401 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5748.9 (0.17 ms/张) |
| End-to-End FPS | 227.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9755 | 0.9561 | 0.9492 | 0.9677 | 0.9924 | 0.9872 | 208 |
| Malignant_lesions | 0.9716 | 0.9725 | 0.9396 | 0.9553 | 0.9873 | 0.9941 | 0.9848 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9922 | 0.9841 | 0.9764 | 0.9845 | 0.9964 | 0.9956 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=9 TN=359 FN=12
- **Malignant_lesions**: TP=171 FP=11 TN=389 FN=5
- **body_of_stomach_normal**: TP=186 FP=3 TN=381 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 10 | 2 |
| Malignant_lesions | 4 | 171 | 1 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9561    0.9423    0.9492       208
     Malignant_lesions     0.9396    0.9716    0.9553       176
body_of_stomach_normal     0.9841    0.9688    0.9764       192

              accuracy                         0.9601       576
             macro avg     0.9599    0.9609    0.9603       576
          weighted avg     0.9604    0.9601    0.9601       576

```