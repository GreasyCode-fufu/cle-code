# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce4s_seed42/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9688 |
| Accuracy | 0.9688 |
| Macro-AUROC (OvR) | 0.9964 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9687 |
| Balanced Acc | 0.9691 |
| Macro-Sensitivity | 0.9691 |
| Macro-Specificity | 0.9844 |
| Macro-Precision | 0.9685 |
| AUC-PR (Macro) | 0.9929 |
| Cohen's Kappa | 0.9530 |
| MCC | 0.9531 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5434.7 (0.18 ms/张) |
| End-to-End FPS | 14.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9810 | 0.9659 | 0.9588 | 0.9730 | 0.9939 | 0.9913 | 208 |
| Malignant_lesions | 0.9659 | 0.9800 | 0.9551 | 0.9605 | 0.9849 | 0.9959 | 0.9889 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9922 | 0.9845 | 0.9870 | 0.9948 | 0.9993 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=7 TN=361 FN=10
- **Malignant_lesions**: TP=170 FP=8 TN=392 FN=6
- **body_of_stomach_normal**: TP=190 FP=3 TN=381 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 8 | 2 |
| Malignant_lesions | 5 | 170 | 1 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9659    0.9519    0.9588       208
     Malignant_lesions     0.9551    0.9659    0.9605       176
body_of_stomach_normal     0.9845    0.9896    0.9870       192

              accuracy                         0.9688       576
             macro avg     0.9685    0.9691    0.9688       576
          weighted avg     0.9688    0.9688    0.9687       576

```