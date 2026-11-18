# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce4s_seed43/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9638 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9962 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9636 |
| Balanced Acc | 0.9638 |
| Macro-Sensitivity | 0.9638 |
| Macro-Specificity | 0.9817 |
| Macro-Precision | 0.9638 |
| AUC-PR (Macro) | 0.9927 |
| Cohen's Kappa | 0.9452 |
| MCC | 0.9452 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5810.3 (0.17 ms/张) |
| End-to-End FPS | 223.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9728 | 0.9522 | 0.9544 | 0.9755 | 0.9940 | 0.9910 | 208 |
| Malignant_lesions | 0.9659 | 0.9800 | 0.9551 | 0.9605 | 0.9849 | 0.9960 | 0.9897 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9922 | 0.9841 | 0.9764 | 0.9845 | 0.9986 | 0.9974 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=10 TN=358 FN=9
- **Malignant_lesions**: TP=170 FP=8 TN=392 FN=6
- **body_of_stomach_normal**: TP=186 FP=3 TN=381 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 5 | 170 | 1 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9522    0.9567    0.9544       208
     Malignant_lesions     0.9551    0.9659    0.9605       176
body_of_stomach_normal     0.9841    0.9688    0.9764       192

              accuracy                         0.9635       576
             macro avg     0.9638    0.9638    0.9638       576
          weighted avg     0.9637    0.9635    0.9636       576

```