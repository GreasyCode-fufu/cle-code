# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_svd_seed43/convnext_svd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9599 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9600 |
| Balanced Acc | 0.9617 |
| Macro-Sensitivity | 0.9617 |
| Macro-Specificity | 0.9804 |
| Macro-Precision | 0.9596 |
| AUC-PR (Macro) | 0.9917 |
| Cohen's Kappa | 0.9401 |
| MCC | 0.9408 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 701.2 (1.43 ms/张) |
| End-to-End FPS | 213.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9183 | 0.9918 | 0.9845 | 0.9502 | 0.9555 | 0.9950 | 0.9927 | 208 |
| Malignant_lesions | 0.9773 | 0.9625 | 0.9198 | 0.9477 | 0.9897 | 0.9938 | 0.9832 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9996 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=3 TN=365 FN=17
- **Malignant_lesions**: TP=172 FP=15 TN=385 FN=4
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 14 | 3 |
| Malignant_lesions | 2 | 172 | 2 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9845    0.9183    0.9502       208
     Malignant_lesions     0.9198    0.9773    0.9477       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9601       576
             macro avg     0.9596    0.9617    0.9599       576
          weighted avg     0.9614    0.9601    0.9600       576

```