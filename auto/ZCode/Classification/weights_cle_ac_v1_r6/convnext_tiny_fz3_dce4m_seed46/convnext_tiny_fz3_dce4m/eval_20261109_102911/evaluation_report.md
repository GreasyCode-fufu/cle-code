# convnext_tiny_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_dce4m_seed46/convnext_tiny_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9530 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9965 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9531 |
| Balanced Acc | 0.9526 |
| Macro-Sensitivity | 0.9526 |
| Macro-Specificity | 0.9764 |
| Macro-Precision | 0.9534 |
| AUC-PR (Macro) | 0.9932 |
| Cohen's Kappa | 0.9295 |
| MCC | 0.9295 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1661.4 (0.60 ms/张) |
| End-to-End FPS | 7.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9647 | 0.9384 | 0.9451 | 0.9726 | 0.9953 | 0.9921 | 208 |
| Malignant_lesions | 0.9318 | 0.9775 | 0.9480 | 0.9398 | 0.9702 | 0.9957 | 0.9903 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9870 | 0.9740 | 0.9740 | 0.9870 | 0.9985 | 0.9972 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=13 TN=355 FN=10
- **Malignant_lesions**: TP=164 FP=9 TN=391 FN=12
- **body_of_stomach_normal**: TP=187 FP=5 TN=379 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 9 | 1 |
| Malignant_lesions | 8 | 164 | 4 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9384    0.9519    0.9451       208
     Malignant_lesions     0.9480    0.9318    0.9398       176
body_of_stomach_normal     0.9740    0.9740    0.9740       192

              accuracy                         0.9531       576
             macro avg     0.9534    0.9526    0.9530       576
          weighted avg     0.9532    0.9531    0.9531       576

```