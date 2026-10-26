# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/ghostnet_seed44/ghostnet/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9723 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9982 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9766 |
| Balanced Acc | 0.9754 |
| Macro-Sensitivity | 0.9754 |
| Macro-Specificity | 0.9887 |
| Macro-Precision | 0.9695 |
| AUC-PR (Macro) | 0.9943 |
| Cohen's Kappa | 0.9629 |
| MCC | 0.9630 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 2854.0 (0.35 ms/张) |
| End-to-End FPS | 248.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9641 | 0.9914 | 0.9895 | 0.9766 | 0.9705 | 0.9980 | 0.9977 | 195 |
| Malignant_lesions | 0.9620 | 0.9856 | 0.9383 | 0.9500 | 0.9913 | 0.9967 | 0.9852 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9891 | 0.9808 | 0.9903 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=2 TN=230 FN=7
- **Malignant_lesions**: TP=76 FP=5 TN=343 FN=3
- **body_of_stomach_normal**: TP=153 FP=3 TN=271 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 5 | 2 |
| Malignant_lesions | 2 | 76 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9895    0.9641    0.9766       195
     Malignant_lesions     0.9383    0.9620    0.9500        79
body_of_stomach_normal     0.9808    1.0000    0.9903       153

              accuracy                         0.9766       427
             macro avg     0.9695    0.9754    0.9723       427
          weighted avg     0.9769    0.9766    0.9766       427

```