# convnext_cmp 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_cmp_seed43/convnext_cmp/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9632 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9977 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9635 |
| Balanced Acc | 0.9643 |
| Macro-Sensitivity | 0.9643 |
| Macro-Specificity | 0.9820 |
| Macro-Precision | 0.9627 |
| AUC-PR (Macro) | 0.9954 |
| Cohen's Kappa | 0.9452 |
| MCC | 0.9455 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 458.6 (2.18 ms/张) |
| End-to-End FPS | 188.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9864 | 0.9749 | 0.9533 | 0.9629 | 0.9971 | 0.9953 | 208 |
| Malignant_lesions | 0.9602 | 0.9675 | 0.9286 | 0.9441 | 0.9822 | 0.9962 | 0.9910 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9922 | 0.9846 | 0.9922 | 1.0000 | 0.9999 | 0.9998 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=5 TN=363 FN=14
- **Malignant_lesions**: TP=169 FP=13 TN=387 FN=7
- **body_of_stomach_normal**: TP=192 FP=3 TN=381 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 13 | 1 |
| Malignant_lesions | 5 | 169 | 2 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9749    0.9327    0.9533       208
     Malignant_lesions     0.9286    0.9602    0.9441       176
body_of_stomach_normal     0.9846    1.0000    0.9922       192

              accuracy                         0.9635       576
             macro avg     0.9627    0.9643    0.9632       576
          weighted avg     0.9640    0.9635    0.9635       576

```