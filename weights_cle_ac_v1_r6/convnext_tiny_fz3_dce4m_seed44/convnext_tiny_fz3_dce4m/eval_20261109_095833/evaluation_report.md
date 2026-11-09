# convnext_tiny_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_dce4m_seed44/convnext_tiny_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9545 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9965 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9547 |
| Balanced Acc | 0.9541 |
| Macro-Sensitivity | 0.9541 |
| Macro-Specificity | 0.9773 |
| Macro-Precision | 0.9551 |
| AUC-PR (Macro) | 0.9933 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9323 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1494.2 (0.67 ms/张) |
| End-to-End FPS | 22.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9728 | 0.9519 | 0.9519 | 0.9728 | 0.9949 | 0.9914 | 208 |
| Malignant_lesions | 0.9261 | 0.9825 | 0.9588 | 0.9422 | 0.9680 | 0.9954 | 0.9900 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9766 | 0.9545 | 0.9692 | 0.9921 | 0.9992 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=10 TN=358 FN=10
- **Malignant_lesions**: TP=163 FP=7 TN=393 FN=13
- **body_of_stomach_normal**: TP=189 FP=9 TN=375 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 7 | 163 | 6 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9519    0.9519    0.9519       208
     Malignant_lesions     0.9588    0.9261    0.9422       176
body_of_stomach_normal     0.9545    0.9844    0.9692       192

              accuracy                         0.9549       576
             macro avg     0.9551    0.9541    0.9545       576
          weighted avg     0.9549    0.9549    0.9547       576

```