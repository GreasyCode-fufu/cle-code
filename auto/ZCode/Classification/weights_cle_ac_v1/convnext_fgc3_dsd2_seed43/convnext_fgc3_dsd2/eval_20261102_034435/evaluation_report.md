# convnext_fgc3_dsd2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc3_dsd2_seed43/convnext_fgc3_dsd2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9336 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9921 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9338 |
| Balanced Acc | 0.9333 |
| Macro-Sensitivity | 0.9333 |
| Macro-Specificity | 0.9668 |
| Macro-Precision | 0.9341 |
| AUC-PR (Macro) | 0.9848 |
| Cohen's Kappa | 0.9008 |
| MCC | 0.9009 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.41G |
| FLOPs | 8.81G |
| Params | 27.61M |
| Pure-Inference FPS | 361.6 (2.77 ms/张) |
| End-to-End FPS | 168.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9538 | 0.9187 | 0.9209 | 0.9564 | 0.9886 | 0.9818 | 208 |
| Malignant_lesions | 0.8977 | 0.9700 | 0.9294 | 0.9133 | 0.9557 | 0.9892 | 0.9756 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9766 | 0.9543 | 0.9666 | 0.9894 | 0.9984 | 0.9969 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=17 TN=351 FN=16
- **Malignant_lesions**: TP=158 FP=12 TN=388 FN=18
- **body_of_stomach_normal**: TP=188 FP=9 TN=375 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 12 | 4 |
| Malignant_lesions | 13 | 158 | 5 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9187    0.9231    0.9209       208
     Malignant_lesions     0.9294    0.8977    0.9133       176
body_of_stomach_normal     0.9543    0.9792    0.9666       192

              accuracy                         0.9340       576
             macro avg     0.9341    0.9333    0.9336       576
          weighted avg     0.9338    0.9340    0.9338       576

```