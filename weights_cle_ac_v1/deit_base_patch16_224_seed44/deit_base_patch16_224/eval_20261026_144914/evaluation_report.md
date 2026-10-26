# deit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/deit_base_patch16_224_seed44/deit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9567 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9941 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9566 |
| Balanced Acc | 0.9565 |
| Macro-Sensitivity | 0.9565 |
| Macro-Specificity | 0.9782 |
| Macro-Precision | 0.9569 |
| AUC-PR (Macro) | 0.9889 |
| Cohen's Kappa | 0.9347 |
| MCC | 0.9348 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 342.2 (2.92 ms/张) |
| End-to-End FPS | 159.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9674 | 0.9429 | 0.9474 | 0.9727 | 0.9904 | 0.9881 | 208 |
| Malignant_lesions | 0.9489 | 0.9775 | 0.9489 | 0.9489 | 0.9775 | 0.9925 | 0.9798 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9896 | 0.9789 | 0.9738 | 0.9845 | 0.9994 | 0.9988 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=12 TN=356 FN=10
- **Malignant_lesions**: TP=167 FP=9 TN=391 FN=9
- **body_of_stomach_normal**: TP=186 FP=4 TN=380 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 8 | 2 |
| Malignant_lesions | 7 | 167 | 2 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9429    0.9519    0.9474       208
     Malignant_lesions     0.9489    0.9489    0.9489       176
body_of_stomach_normal     0.9789    0.9688    0.9738       192

              accuracy                         0.9566       576
             macro avg     0.9569    0.9565    0.9567       576
          weighted avg     0.9567    0.9566    0.9566       576

```