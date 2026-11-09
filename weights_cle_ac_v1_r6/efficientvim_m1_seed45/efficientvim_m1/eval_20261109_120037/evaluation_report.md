# efficientvim_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/efficientvim_m1_seed45/efficientvim_m1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9464 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9930 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9463 |
| Balanced Acc | 0.9446 |
| Macro-Sensitivity | 0.9446 |
| Macro-Specificity | 0.9725 |
| Macro-Precision | 0.9496 |
| AUC-PR (Macro) | 0.9868 |
| Cohen's Kappa | 0.9189 |
| MCC | 0.9196 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 234.46M |
| FLOPs | 468.92M |
| Params | 5.71M |
| Pure-Inference FPS | 7359.8 (0.14 ms/张) |
| End-to-End FPS | 290.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9402 | 0.9009 | 0.9302 | 0.9774 | 0.9896 | 0.9828 | 208 |
| Malignant_lesions | 0.9034 | 0.9850 | 0.9636 | 0.9326 | 0.9586 | 0.9907 | 0.9797 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9922 | 0.9841 | 0.9764 | 0.9845 | 0.9989 | 0.9980 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=22 TN=346 FN=8
- **Malignant_lesions**: TP=159 FP=6 TN=394 FN=17
- **body_of_stomach_normal**: TP=186 FP=3 TN=381 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 16 | 159 | 1 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9009    0.9615    0.9302       208
     Malignant_lesions     0.9636    0.9034    0.9326       176
body_of_stomach_normal     0.9841    0.9688    0.9764       192

              accuracy                         0.9462       576
             macro avg     0.9496    0.9446    0.9464       576
          weighted avg     0.9478    0.9462    0.9463       576

```