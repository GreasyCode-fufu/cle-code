# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_seed46/fast_t0/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9289 |
| Accuracy | 0.9288 |
| Macro-AUROC (OvR) | 0.9916 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9290 |
| Balanced Acc | 0.9260 |
| Macro-Sensitivity | 0.9260 |
| Macro-Specificity | 0.9634 |
| Macro-Precision | 0.9356 |
| AUC-PR (Macro) | 0.9838 |
| Cohen's Kappa | 0.8927 |
| MCC | 0.8946 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 7576.7 (0.13 ms/张) |
| End-to-End FPS | 86.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9158 | 0.8664 | 0.9136 | 0.9797 | 0.9878 | 0.9800 | 208 |
| Malignant_lesions | 0.8636 | 0.9850 | 0.9620 | 0.9102 | 0.9426 | 0.9890 | 0.9751 | 176 |
| body_of_stomach_normal | 0.9479 | 0.9896 | 0.9785 | 0.9630 | 0.9744 | 0.9981 | 0.9964 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=31 TN=337 FN=7
- **Malignant_lesions**: TP=152 FP=6 TN=394 FN=24
- **body_of_stomach_normal**: TP=182 FP=4 TN=380 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 21 | 152 | 3 |
| body_of_stomach_normal | 10 | 0 | 182 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8664    0.9663    0.9136       208
     Malignant_lesions     0.9620    0.8636    0.9102       176
body_of_stomach_normal     0.9785    0.9479    0.9630       192

              accuracy                         0.9288       576
             macro avg     0.9356    0.9260    0.9289       576
          weighted avg     0.9330    0.9288    0.9290       576

```