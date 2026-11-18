# mbv3_r6_s2_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_fz16_seed44/mbv3_r6_s2_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9550 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9956 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9548 |
| Balanced Acc | 0.9537 |
| Macro-Sensitivity | 0.9537 |
| Macro-Specificity | 0.9770 |
| Macro-Precision | 0.9570 |
| AUC-PR (Macro) | 0.9916 |
| Cohen's Kappa | 0.9320 |
| MCC | 0.9324 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 4393.8 (0.23 ms/张) |
| End-to-End FPS | 237.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9565 | 0.9259 | 0.9434 | 0.9778 | 0.9925 | 0.9887 | 208 |
| Malignant_lesions | 0.9205 | 0.9900 | 0.9759 | 0.9474 | 0.9659 | 0.9952 | 0.9879 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9844 | 0.9691 | 0.9741 | 0.9895 | 0.9990 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=16 TN=352 FN=8
- **Malignant_lesions**: TP=162 FP=4 TN=396 FN=14
- **body_of_stomach_normal**: TP=188 FP=6 TN=378 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 4 | 4 |
| Malignant_lesions | 12 | 162 | 2 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9259    0.9615    0.9434       208
     Malignant_lesions     0.9759    0.9205    0.9474       176
body_of_stomach_normal     0.9691    0.9792    0.9741       192

              accuracy                         0.9549       576
             macro avg     0.9570    0.9537    0.9550       576
          weighted avg     0.9556    0.9549    0.9548       576

```