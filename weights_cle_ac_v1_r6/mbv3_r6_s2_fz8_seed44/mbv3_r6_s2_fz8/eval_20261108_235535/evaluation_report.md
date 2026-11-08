# mbv3_r6_s2_fz8 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_fz8_seed44/mbv3_r6_s2_fz8/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9465 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9920 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9464 |
| Balanced Acc | 0.9446 |
| Macro-Sensitivity | 0.9446 |
| Macro-Specificity | 0.9724 |
| Macro-Precision | 0.9500 |
| AUC-PR (Macro) | 0.9855 |
| Cohen's Kappa | 0.9189 |
| MCC | 0.9197 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 4410.6 (0.23 ms/张) |
| End-to-End FPS | 187.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9375 | 0.8969 | 0.9281 | 0.9773 | 0.9884 | 0.9779 | 208 |
| Malignant_lesions | 0.9034 | 0.9850 | 0.9636 | 0.9326 | 0.9586 | 0.9883 | 0.9800 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9948 | 0.9894 | 0.9789 | 0.9845 | 0.9992 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=23 TN=345 FN=8
- **Malignant_lesions**: TP=159 FP=6 TN=394 FN=17
- **body_of_stomach_normal**: TP=186 FP=2 TN=382 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 17 | 159 | 0 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8969    0.9615    0.9281       208
     Malignant_lesions     0.9636    0.9034    0.9326       176
body_of_stomach_normal     0.9894    0.9688    0.9789       192

              accuracy                         0.9462       576
             macro avg     0.9500    0.9446    0.9465       576
          weighted avg     0.9481    0.9462    0.9464       576

```