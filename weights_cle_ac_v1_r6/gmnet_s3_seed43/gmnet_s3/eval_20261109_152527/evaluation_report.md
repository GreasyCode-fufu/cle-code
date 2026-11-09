# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/gmnet_s3_seed43/gmnet_s3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9355 |
| Accuracy | 0.9358 |
| Macro-AUROC (OvR) | 0.9919 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9357 |
| Balanced Acc | 0.9337 |
| Macro-Sensitivity | 0.9337 |
| Macro-Specificity | 0.9673 |
| Macro-Precision | 0.9390 |
| AUC-PR (Macro) | 0.9848 |
| Cohen's Kappa | 0.9032 |
| MCC | 0.9040 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 878.7 (1.14 ms/张) |
| End-to-End FPS | 215.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9375 | 0.8969 | 0.9281 | 0.9773 | 0.9884 | 0.9807 | 208 |
| Malignant_lesions | 0.8864 | 0.9825 | 0.9571 | 0.9204 | 0.9516 | 0.9900 | 0.9786 | 176 |
| body_of_stomach_normal | 0.9531 | 0.9818 | 0.9632 | 0.9581 | 0.9767 | 0.9972 | 0.9952 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=23 TN=345 FN=8
- **Malignant_lesions**: TP=156 FP=7 TN=393 FN=20
- **body_of_stomach_normal**: TP=183 FP=7 TN=377 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 15 | 156 | 5 |
| body_of_stomach_normal | 8 | 1 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8969    0.9615    0.9281       208
     Malignant_lesions     0.9571    0.8864    0.9204       176
body_of_stomach_normal     0.9632    0.9531    0.9581       192

              accuracy                         0.9358       576
             macro avg     0.9390    0.9337    0.9355       576
          weighted avg     0.9374    0.9358    0.9357       576

```