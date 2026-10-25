# convnext_wds_wtb 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wtb_seed42/convnext_wds_wtb/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9550 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9952 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9550 |
| Balanced Acc | 0.9543 |
| Macro-Sensitivity | 0.9543 |
| Macro-Specificity | 0.9772 |
| Macro-Precision | 0.9558 |
| AUC-PR (Macro) | 0.9910 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9322 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.53G |
| FLOPs | 9.07G |
| Params | 29.11M |
| Pure-Inference FPS | 526.5 (1.90 ms/张) |
| End-to-End FPS | 190.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9592 | 0.9299 | 0.9431 | 0.9751 | 0.9931 | 0.9879 | 208 |
| Malignant_lesions | 0.9375 | 0.9775 | 0.9483 | 0.9429 | 0.9726 | 0.9937 | 0.9873 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9948 | 0.9894 | 0.9789 | 0.9845 | 0.9987 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=15 TN=353 FN=9
- **Malignant_lesions**: TP=165 FP=9 TN=391 FN=11
- **body_of_stomach_normal**: TP=186 FP=2 TN=382 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 10 | 165 | 1 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9299    0.9567    0.9431       208
     Malignant_lesions     0.9483    0.9375    0.9429       176
body_of_stomach_normal     0.9894    0.9688    0.9789       192

              accuracy                         0.9549       576
             macro avg     0.9558    0.9543    0.9550       576
          weighted avg     0.9553    0.9549    0.9550       576

```