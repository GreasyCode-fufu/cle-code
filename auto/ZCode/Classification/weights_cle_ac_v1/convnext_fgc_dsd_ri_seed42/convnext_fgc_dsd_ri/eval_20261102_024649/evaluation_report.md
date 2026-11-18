# convnext_fgc_dsd_ri 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_dsd_ri_seed42/convnext_fgc_dsd_ri/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9542 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9946 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9546 |
| Balanced Acc | 0.9531 |
| Macro-Sensitivity | 0.9531 |
| Macro-Specificity | 0.9770 |
| Macro-Precision | 0.9563 |
| AUC-PR (Macro) | 0.9897 |
| Cohen's Kappa | 0.9320 |
| MCC | 0.9325 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 452.3 (2.21 ms/张) |
| End-to-End FPS | 182.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9565 | 0.9259 | 0.9434 | 0.9778 | 0.9917 | 0.9868 | 208 |
| Malignant_lesions | 0.8977 | 0.9850 | 0.9634 | 0.9294 | 0.9563 | 0.9925 | 0.9828 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9896 | 0.9796 | 0.9897 | 1.0000 | 0.9997 | 0.9994 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=16 TN=352 FN=8
- **Malignant_lesions**: TP=158 FP=6 TN=394 FN=18
- **body_of_stomach_normal**: TP=192 FP=4 TN=380 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 16 | 158 | 2 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9259    0.9615    0.9434       208
     Malignant_lesions     0.9634    0.8977    0.9294       176
body_of_stomach_normal     0.9796    1.0000    0.9897       192

              accuracy                         0.9549       576
             macro avg     0.9563    0.9531    0.9542       576
          weighted avg     0.9553    0.9549    0.9546       576

```