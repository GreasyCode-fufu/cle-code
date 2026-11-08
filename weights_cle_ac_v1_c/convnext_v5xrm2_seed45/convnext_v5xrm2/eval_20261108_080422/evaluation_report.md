# convnext_v5xrm2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5xrm2_seed45/convnext_v5xrm2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9605 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9958 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9602 |
| Balanced Acc | 0.9612 |
| Macro-Sensitivity | 0.9612 |
| Macro-Specificity | 0.9801 |
| Macro-Precision | 0.9602 |
| AUC-PR (Macro) | 0.9920 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9402 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.69M |
| Pure-Inference FPS | 459.5 (2.18 ms/张) |
| End-to-End FPS | 183.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9728 | 0.9512 | 0.9443 | 0.9650 | 0.9939 | 0.9902 | 208 |
| Malignant_lesions | 0.9773 | 0.9700 | 0.9348 | 0.9556 | 0.9898 | 0.9952 | 0.9886 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9974 | 0.9947 | 0.9815 | 0.9846 | 0.9984 | 0.9973 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=10 TN=358 FN=13
- **Malignant_lesions**: TP=172 FP=12 TN=388 FN=4
- **body_of_stomach_normal**: TP=186 FP=1 TN=383 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 12 | 1 |
| Malignant_lesions | 4 | 172 | 0 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9512    0.9375    0.9443       208
     Malignant_lesions     0.9348    0.9773    0.9556       176
body_of_stomach_normal     0.9947    0.9688    0.9815       192

              accuracy                         0.9601       576
             macro avg     0.9602    0.9612    0.9605       576
          weighted avg     0.9607    0.9601    0.9602       576

```