# convnext_dph 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tmp/convnext_dph_seed42/convnext_dph/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9542 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9966 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9546 |
| Balanced Acc | 0.9530 |
| Macro-Sensitivity | 0.9530 |
| Macro-Specificity | 0.9770 |
| Macro-Precision | 0.9565 |
| AUC-PR (Macro) | 0.9933 |
| Cohen's Kappa | 0.9320 |
| MCC | 0.9325 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 380.5 (2.63 ms/张) |
| End-to-End FPS | 172.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9592 | 0.9309 | 0.9506 | 0.9833 | 0.9950 | 0.9922 | 208 |
| Malignant_lesions | 0.9034 | 0.9875 | 0.9695 | 0.9353 | 0.9587 | 0.9957 | 0.9896 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9844 | 0.9692 | 0.9767 | 0.9921 | 0.9990 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=15 TN=353 FN=6
- **Malignant_lesions**: TP=159 FP=5 TN=395 FN=17
- **body_of_stomach_normal**: TP=189 FP=6 TN=378 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 12 | 159 | 5 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9309    0.9712    0.9506       208
     Malignant_lesions     0.9695    0.9034    0.9353       176
body_of_stomach_normal     0.9692    0.9844    0.9767       192

              accuracy                         0.9549       576
             macro avg     0.9565    0.9530    0.9542       576
          weighted avg     0.9555    0.9549    0.9546       576

```