# mbv3_r6_star2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star2_seed44/mbv3_r6_star2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9342 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9924 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9345 |
| Balanced Acc | 0.9340 |
| Macro-Sensitivity | 0.9340 |
| Macro-Specificity | 0.9670 |
| Macro-Precision | 0.9349 |
| AUC-PR (Macro) | 0.9861 |
| Cohen's Kappa | 0.9008 |
| MCC | 0.9011 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 3832.4 (0.26 ms/张) |
| End-to-End FPS | 163.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9511 | 0.9147 | 0.9212 | 0.9589 | 0.9893 | 0.9821 | 208 |
| Malignant_lesions | 0.9261 | 0.9525 | 0.8956 | 0.9106 | 0.9670 | 0.9885 | 0.9776 | 176 |
| body_of_stomach_normal | 0.9479 | 0.9974 | 0.9945 | 0.9707 | 0.9746 | 0.9994 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=18 TN=350 FN=15
- **Malignant_lesions**: TP=163 FP=19 TN=381 FN=13
- **body_of_stomach_normal**: TP=182 FP=1 TN=383 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 14 | 1 |
| Malignant_lesions | 13 | 163 | 0 |
| body_of_stomach_normal | 5 | 5 | 182 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9147    0.9279    0.9212       208
     Malignant_lesions     0.8956    0.9261    0.9106       176
body_of_stomach_normal     0.9945    0.9479    0.9707       192

              accuracy                         0.9340       576
             macro avg     0.9349    0.9340    0.9342       576
          weighted avg     0.9355    0.9340    0.9345       576

```