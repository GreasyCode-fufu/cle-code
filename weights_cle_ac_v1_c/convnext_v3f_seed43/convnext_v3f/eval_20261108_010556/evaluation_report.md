# convnext_v3f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v3f_seed43/convnext_v3f/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9456 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9950 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9460 |
| Balanced Acc | 0.9442 |
| Macro-Sensitivity | 0.9442 |
| Macro-Specificity | 0.9726 |
| Macro-Precision | 0.9484 |
| AUC-PR (Macro) | 0.9900 |
| Cohen's Kappa | 0.9189 |
| MCC | 0.9196 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.65G |
| FLOPs | 9.31G |
| Params | 28.73M |
| Pure-Inference FPS | 416.6 (2.40 ms/张) |
| End-to-End FPS | 173.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9457 | 0.9087 | 0.9321 | 0.9748 | 0.9927 | 0.9889 | 208 |
| Malignant_lesions | 0.8864 | 0.9825 | 0.9571 | 0.9204 | 0.9516 | 0.9926 | 0.9821 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9896 | 0.9794 | 0.9845 | 0.9948 | 0.9995 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=20 TN=348 FN=9
- **Malignant_lesions**: TP=156 FP=7 TN=393 FN=20
- **body_of_stomach_normal**: TP=190 FP=4 TN=380 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 18 | 156 | 2 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9087    0.9567    0.9321       208
     Malignant_lesions     0.9571    0.8864    0.9204       176
body_of_stomach_normal     0.9794    0.9896    0.9845       192

              accuracy                         0.9462       576
             macro avg     0.9484    0.9442    0.9456       576
          weighted avg     0.9470    0.9462    0.9460       576

```