# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_small_seed44/convnext_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9594 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9936 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9598 |
| Balanced Acc | 0.9586 |
| Macro-Sensitivity | 0.9586 |
| Macro-Specificity | 0.9798 |
| Macro-Precision | 0.9607 |
| AUC-PR (Macro) | 0.9886 |
| Cohen's Kappa | 0.9399 |
| MCC | 0.9402 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.41M |
| Pure-Inference FPS | 486.4 (2.06 ms/张) |
| End-to-End FPS | 183.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9674 | 0.9437 | 0.9549 | 0.9807 | 0.9929 | 0.9879 | 208 |
| Malignant_lesions | 0.9148 | 0.9850 | 0.9641 | 0.9388 | 0.9633 | 0.9889 | 0.9799 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9870 | 0.9745 | 0.9845 | 0.9974 | 0.9989 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=12 TN=356 FN=7
- **Malignant_lesions**: TP=161 FP=6 TN=394 FN=15
- **body_of_stomach_normal**: TP=191 FP=5 TN=379 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 11 | 161 | 4 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9437    0.9663    0.9549       208
     Malignant_lesions     0.9641    0.9148    0.9388       176
body_of_stomach_normal     0.9745    0.9948    0.9845       192

              accuracy                         0.9601       576
             macro avg     0.9607    0.9586    0.9594       576
          weighted avg     0.9602    0.9601    0.9598       576

```