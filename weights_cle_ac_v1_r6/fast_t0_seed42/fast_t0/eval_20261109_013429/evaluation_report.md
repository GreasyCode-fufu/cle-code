# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_seed42/fast_t0/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9335 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9905 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9339 |
| Balanced Acc | 0.9309 |
| Macro-Sensitivity | 0.9309 |
| Macro-Specificity | 0.9661 |
| Macro-Precision | 0.9399 |
| AUC-PR (Macro) | 0.9807 |
| Cohen's Kappa | 0.9005 |
| MCC | 0.9023 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 9314.5 (0.11 ms/张) |
| End-to-End FPS | 145.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9212 | 0.8739 | 0.9178 | 0.9798 | 0.9858 | 0.9774 | 208 |
| Malignant_lesions | 0.8523 | 0.9850 | 0.9615 | 0.9036 | 0.9381 | 0.9867 | 0.9665 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9922 | 0.9842 | 0.9791 | 0.9870 | 0.9991 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=29 TN=339 FN=7
- **Malignant_lesions**: TP=150 FP=6 TN=394 FN=26
- **body_of_stomach_normal**: TP=187 FP=3 TN=381 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 24 | 150 | 2 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8739    0.9663    0.9178       208
     Malignant_lesions     0.9615    0.8523    0.9036       176
body_of_stomach_normal     0.9842    0.9740    0.9791       192

              accuracy                         0.9340       576
             macro avg     0.9399    0.9309    0.9335       576
          weighted avg     0.9375    0.9340    0.9339       576

```