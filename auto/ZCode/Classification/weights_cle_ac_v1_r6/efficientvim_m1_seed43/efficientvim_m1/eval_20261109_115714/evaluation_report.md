# efficientvim_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/efficientvim_m1_seed43/efficientvim_m1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9319 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9952 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9329 |
| Balanced Acc | 0.9305 |
| Macro-Sensitivity | 0.9305 |
| Macro-Specificity | 0.9664 |
| Macro-Precision | 0.9375 |
| AUC-PR (Macro) | 0.9907 |
| Cohen's Kappa | 0.9006 |
| MCC | 0.9023 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 234.46M |
| FLOPs | 468.92M |
| Params | 5.71M |
| Pure-Inference FPS | 5342.0 (0.19 ms/张) |
| End-to-End FPS | 231.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9457 | 0.9091 | 0.9346 | 0.9775 | 0.9926 | 0.9875 | 208 |
| Malignant_lesions | 0.8352 | 0.9875 | 0.9671 | 0.8963 | 0.9316 | 0.9935 | 0.9857 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9661 | 0.9363 | 0.9646 | 0.9973 | 0.9994 | 0.9989 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=20 TN=348 FN=8
- **Malignant_lesions**: TP=147 FP=5 TN=395 FN=29
- **body_of_stomach_normal**: TP=191 FP=13 TN=371 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 5 | 3 |
| Malignant_lesions | 19 | 147 | 10 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9091    0.9615    0.9346       208
     Malignant_lesions     0.9671    0.8352    0.8963       176
body_of_stomach_normal     0.9363    0.9948    0.9646       192

              accuracy                         0.9340       576
             macro avg     0.9375    0.9305    0.9319       576
          weighted avg     0.9359    0.9340    0.9329       576

```