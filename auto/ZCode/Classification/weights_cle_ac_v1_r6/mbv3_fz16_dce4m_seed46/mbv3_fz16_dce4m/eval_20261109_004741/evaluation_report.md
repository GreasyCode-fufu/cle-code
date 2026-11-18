# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce4m_seed46/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9670 |
| Accuracy | 0.9670 |
| Macro-AUROC (OvR) | 0.9960 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9670 |
| Balanced Acc | 0.9673 |
| Macro-Sensitivity | 0.9673 |
| Macro-Specificity | 0.9835 |
| Macro-Precision | 0.9667 |
| AUC-PR (Macro) | 0.9922 |
| Cohen's Kappa | 0.9504 |
| MCC | 0.9504 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5640.9 (0.18 ms/张) |
| End-to-End FPS | 270.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9810 | 0.9660 | 0.9614 | 0.9757 | 0.9937 | 0.9910 | 208 |
| Malignant_lesions | 0.9659 | 0.9800 | 0.9551 | 0.9605 | 0.9849 | 0.9953 | 0.9876 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9896 | 0.9792 | 0.9792 | 0.9896 | 0.9990 | 0.9980 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=7 TN=361 FN=9
- **Malignant_lesions**: TP=170 FP=8 TN=392 FN=6
- **body_of_stomach_normal**: TP=188 FP=4 TN=380 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 4 | 170 | 2 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9660    0.9567    0.9614       208
     Malignant_lesions     0.9551    0.9659    0.9605       176
body_of_stomach_normal     0.9792    0.9792    0.9792       192

              accuracy                         0.9670       576
             macro avg     0.9667    0.9673    0.9670       576
          weighted avg     0.9671    0.9670    0.9670       576

```