# shvit_s1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/shvit_s1_seed45/shvit_s1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9321 |
| Accuracy | 0.9323 |
| Macro-AUROC (OvR) | 0.9890 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9322 |
| Balanced Acc | 0.9319 |
| Macro-Sensitivity | 0.9319 |
| Macro-Specificity | 0.9659 |
| Macro-Precision | 0.9324 |
| AUC-PR (Macro) | 0.9787 |
| Cohen's Kappa | 0.8982 |
| MCC | 0.8982 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 245.71M |
| FLOPs | 491.41M |
| Params | 6.01M |
| Pure-Inference FPS | 4442.7 (0.23 ms/张) |
| End-to-End FPS | 226.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9183 | 0.9511 | 0.9139 | 0.9161 | 0.9537 | 0.9838 | 0.9729 | 208 |
| Malignant_lesions | 0.9034 | 0.9675 | 0.9244 | 0.9138 | 0.9579 | 0.9850 | 0.9665 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9792 | 0.9590 | 0.9664 | 0.9869 | 0.9981 | 0.9967 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=18 TN=350 FN=17
- **Malignant_lesions**: TP=159 FP=13 TN=387 FN=17
- **body_of_stomach_normal**: TP=187 FP=8 TN=376 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 13 | 4 |
| Malignant_lesions | 13 | 159 | 4 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9139    0.9183    0.9161       208
     Malignant_lesions     0.9244    0.9034    0.9138       176
body_of_stomach_normal     0.9590    0.9740    0.9664       192

              accuracy                         0.9323       576
             macro avg     0.9324    0.9319    0.9321       576
          weighted avg     0.9321    0.9323    0.9322       576

```