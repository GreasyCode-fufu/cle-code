# mbv3_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce4s_seed43/mbv3_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9583 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9934 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9584 |
| Balanced Acc | 0.9578 |
| Macro-Sensitivity | 0.9578 |
| Macro-Specificity | 0.9790 |
| Macro-Precision | 0.9589 |
| AUC-PR (Macro) | 0.9873 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9374 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3529.8 (0.28 ms/张) |
| End-to-End FPS | 10.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9647 | 0.9387 | 0.9476 | 0.9753 | 0.9899 | 0.9822 | 208 |
| Malignant_lesions | 0.9375 | 0.9800 | 0.9538 | 0.9456 | 0.9727 | 0.9908 | 0.9807 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=13 TN=355 FN=9
- **Malignant_lesions**: TP=165 FP=8 TN=392 FN=11
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 10 | 165 | 1 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9387    0.9567    0.9476       208
     Malignant_lesions     0.9538    0.9375    0.9456       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9583       576
             macro avg     0.9589    0.9578    0.9583       576
          weighted avg     0.9585    0.9583    0.9584       576

```