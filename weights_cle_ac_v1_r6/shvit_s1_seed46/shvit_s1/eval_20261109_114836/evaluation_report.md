# shvit_s1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/shvit_s1_seed46/shvit_s1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9344 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9938 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9342 |
| Balanced Acc | 0.9322 |
| Macro-Sensitivity | 0.9322 |
| Macro-Specificity | 0.9662 |
| Macro-Precision | 0.9385 |
| AUC-PR (Macro) | 0.9886 |
| Cohen's Kappa | 0.9006 |
| MCC | 0.9015 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 245.71M |
| FLOPs | 491.41M |
| Params | 6.01M |
| Pure-Inference FPS | 5442.4 (0.18 ms/张) |
| End-to-End FPS | 252.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9321 | 0.8884 | 0.9213 | 0.9744 | 0.9906 | 0.9849 | 208 |
| Malignant_lesions | 0.8920 | 0.9875 | 0.9691 | 0.9290 | 0.9541 | 0.9936 | 0.9859 | 176 |
| body_of_stomach_normal | 0.9479 | 0.9792 | 0.9579 | 0.9529 | 0.9741 | 0.9973 | 0.9949 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=25 TN=343 FN=9
- **Malignant_lesions**: TP=157 FP=5 TN=395 FN=19
- **body_of_stomach_normal**: TP=182 FP=8 TN=376 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 4 | 5 |
| Malignant_lesions | 16 | 157 | 3 |
| body_of_stomach_normal | 9 | 1 | 182 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8884    0.9567    0.9213       208
     Malignant_lesions     0.9691    0.8920    0.9290       176
body_of_stomach_normal     0.9579    0.9479    0.9529       192

              accuracy                         0.9340       576
             macro avg     0.9385    0.9322    0.9344       576
          weighted avg     0.9362    0.9340    0.9342       576

```