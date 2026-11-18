# mbv3_r6_star3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star3_seed43/mbv3_r6_star3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9456 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9922 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9459 |
| Balanced Acc | 0.9445 |
| Macro-Sensitivity | 0.9445 |
| Macro-Specificity | 0.9727 |
| Macro-Precision | 0.9475 |
| AUC-PR (Macro) | 0.9853 |
| Cohen's Kappa | 0.9190 |
| MCC | 0.9194 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.97M |
| FLOPs | 463.94M |
| Params | 3.26M |
| Pure-Inference FPS | 5610.9 (0.18 ms/张) |
| End-to-End FPS | 297.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9511 | 0.9167 | 0.9340 | 0.9722 | 0.9885 | 0.9796 | 208 |
| Malignant_lesions | 0.8920 | 0.9800 | 0.9515 | 0.9208 | 0.9538 | 0.9890 | 0.9778 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9992 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=18 TN=350 FN=10
- **Malignant_lesions**: TP=157 FP=8 TN=392 FN=19
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 8 | 2 |
| Malignant_lesions | 16 | 157 | 3 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9167    0.9519    0.9340       208
     Malignant_lesions     0.9515    0.8920    0.9208       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9462       576
             macro avg     0.9475    0.9445    0.9456       576
          weighted avg     0.9465    0.9462    0.9459       576

```