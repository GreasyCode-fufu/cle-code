# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_dsd_seed43/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9563 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9948 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9564 |
| Balanced Acc | 0.9557 |
| Macro-Sensitivity | 0.9557 |
| Macro-Specificity | 0.9781 |
| Macro-Precision | 0.9572 |
| AUC-PR (Macro) | 0.9900 |
| Cohen's Kappa | 0.9347 |
| MCC | 0.9349 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 1708.8 (0.59 ms/张) |
| End-to-End FPS | 271.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9647 | 0.9384 | 0.9451 | 0.9726 | 0.9929 | 0.9862 | 208 |
| Malignant_lesions | 0.9205 | 0.9825 | 0.9586 | 0.9391 | 0.9656 | 0.9918 | 0.9843 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9870 | 0.9745 | 0.9845 | 0.9974 | 0.9998 | 0.9996 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=13 TN=355 FN=10
- **Malignant_lesions**: TP=162 FP=7 TN=393 FN=14
- **body_of_stomach_normal**: TP=191 FP=5 TN=379 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 12 | 162 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9384    0.9519    0.9451       208
     Malignant_lesions     0.9586    0.9205    0.9391       176
body_of_stomach_normal     0.9745    0.9948    0.9845       192

              accuracy                         0.9566       576
             macro avg     0.9572    0.9557    0.9563       576
          weighted avg     0.9566    0.9566    0.9564       576

```