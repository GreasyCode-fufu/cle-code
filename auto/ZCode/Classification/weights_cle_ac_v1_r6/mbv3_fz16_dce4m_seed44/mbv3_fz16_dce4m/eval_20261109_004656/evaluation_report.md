# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce4m_seed44/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9585 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9944 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9584 |
| Balanced Acc | 0.9593 |
| Macro-Sensitivity | 0.9593 |
| Macro-Specificity | 0.9792 |
| Macro-Precision | 0.9581 |
| AUC-PR (Macro) | 0.9896 |
| Cohen's Kappa | 0.9374 |
| MCC | 0.9376 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5627.2 (0.18 ms/张) |
| End-to-End FPS | 220.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9755 | 0.9559 | 0.9466 | 0.9651 | 0.9926 | 0.9878 | 208 |
| Malignant_lesions | 0.9716 | 0.9700 | 0.9344 | 0.9526 | 0.9873 | 0.9941 | 0.9853 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9922 | 0.9841 | 0.9764 | 0.9845 | 0.9965 | 0.9957 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=9 TN=359 FN=13
- **Malignant_lesions**: TP=171 FP=12 TN=388 FN=5
- **body_of_stomach_normal**: TP=186 FP=3 TN=381 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 11 | 2 |
| Malignant_lesions | 4 | 171 | 1 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9559    0.9375    0.9466       208
     Malignant_lesions     0.9344    0.9716    0.9526       176
body_of_stomach_normal     0.9841    0.9688    0.9764       192

              accuracy                         0.9583       576
             macro avg     0.9581    0.9593    0.9585       576
          weighted avg     0.9587    0.9583    0.9584       576

```