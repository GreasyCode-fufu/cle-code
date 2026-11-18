# convnext_dpea 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dpea_seed43/convnext_dpea/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9462 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9950 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9465 |
| Balanced Acc | 0.9459 |
| Macro-Sensitivity | 0.9459 |
| Macro-Specificity | 0.9730 |
| Macro-Precision | 0.9468 |
| AUC-PR (Macro) | 0.9899 |
| Cohen's Kappa | 0.9191 |
| MCC | 0.9193 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 28.23M |
| Pure-Inference FPS | 461.4 (2.17 ms/张) |
| End-to-End FPS | 185.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9592 | 0.9292 | 0.9381 | 0.9698 | 0.9930 | 0.9898 | 208 |
| Malignant_lesions | 0.9375 | 0.9625 | 0.9167 | 0.9270 | 0.9722 | 0.9929 | 0.9816 | 176 |
| body_of_stomach_normal | 0.9531 | 0.9974 | 0.9946 | 0.9734 | 0.9770 | 0.9991 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=15 TN=353 FN=11
- **Malignant_lesions**: TP=165 FP=15 TN=385 FN=11
- **body_of_stomach_normal**: TP=183 FP=1 TN=383 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 11 | 165 | 0 |
| body_of_stomach_normal | 4 | 5 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9292    0.9471    0.9381       208
     Malignant_lesions     0.9167    0.9375    0.9270       176
body_of_stomach_normal     0.9946    0.9531    0.9734       192

              accuracy                         0.9462       576
             macro avg     0.9468    0.9459    0.9462       576
          weighted avg     0.9472    0.9462    0.9465       576

```