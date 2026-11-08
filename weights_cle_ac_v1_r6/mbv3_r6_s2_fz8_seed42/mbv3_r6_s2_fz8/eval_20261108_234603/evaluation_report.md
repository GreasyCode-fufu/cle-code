# mbv3_r6_s2_fz8 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_fz8_seed42/mbv3_r6_s2_fz8/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9553 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9956 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9551 |
| Balanced Acc | 0.9529 |
| Macro-Sensitivity | 0.9529 |
| Macro-Specificity | 0.9767 |
| Macro-Precision | 0.9599 |
| AUC-PR (Macro) | 0.9919 |
| Cohen's Kappa | 0.9320 |
| MCC | 0.9331 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 5037.9 (0.20 ms/张) |
| End-to-End FPS | 236.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9402 | 0.9027 | 0.9401 | 0.9886 | 0.9929 | 0.9873 | 208 |
| Malignant_lesions | 0.9091 | 0.9950 | 0.9877 | 0.9467 | 0.9614 | 0.9958 | 0.9918 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9948 | 0.9894 | 0.9789 | 0.9845 | 0.9980 | 0.9965 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=22 TN=346 FN=4
- **Malignant_lesions**: TP=160 FP=2 TN=398 FN=16
- **body_of_stomach_normal**: TP=186 FP=2 TN=382 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 2 | 2 |
| Malignant_lesions | 16 | 160 | 0 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9027    0.9808    0.9401       208
     Malignant_lesions     0.9877    0.9091    0.9467       176
body_of_stomach_normal     0.9894    0.9688    0.9789       192

              accuracy                         0.9549       576
             macro avg     0.9599    0.9529    0.9553       576
          weighted avg     0.9575    0.9549    0.9551       576

```