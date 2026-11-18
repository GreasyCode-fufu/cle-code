# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/fast_t0_seed43/fast_t0/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9440 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9911 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9444 |
| Balanced Acc | 0.9437 |
| Macro-Sensitivity | 0.9437 |
| Macro-Specificity | 0.9721 |
| Macro-Precision | 0.9445 |
| AUC-PR (Macro) | 0.9828 |
| Cohen's Kappa | 0.9164 |
| MCC | 0.9165 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 3375.4 (0.30 ms/张) |
| End-to-End FPS | 205.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9565 | 0.9242 | 0.9308 | 0.9644 | 0.9878 | 0.9804 | 208 |
| Malignant_lesions | 0.9091 | 0.9675 | 0.9249 | 0.9169 | 0.9603 | 0.9862 | 0.9691 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9922 | 0.9844 | 0.9844 | 0.9922 | 0.9995 | 0.9989 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=16 TN=352 FN=13
- **Malignant_lesions**: TP=160 FP=13 TN=387 FN=16
- **body_of_stomach_normal**: TP=189 FP=3 TN=381 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 12 | 1 |
| Malignant_lesions | 14 | 160 | 2 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9242    0.9375    0.9308       208
     Malignant_lesions     0.9249    0.9091    0.9169       176
body_of_stomach_normal     0.9844    0.9844    0.9844       192

              accuracy                         0.9444       576
             macro avg     0.9445    0.9437    0.9440       576
          weighted avg     0.9444    0.9444    0.9444       576

```