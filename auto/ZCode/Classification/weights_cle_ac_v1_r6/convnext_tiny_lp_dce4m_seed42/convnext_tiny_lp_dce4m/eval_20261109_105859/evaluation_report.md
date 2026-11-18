# convnext_tiny_lp_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_lp_dce4m_seed42/convnext_tiny_lp_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9708 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9990 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9765 |
| Balanced Acc | 0.9704 |
| Macro-Sensitivity | 0.9704 |
| Macro-Specificity | 0.9882 |
| Macro-Precision | 0.9713 |
| AUC-PR (Macro) | 0.9967 |
| Cohen's Kappa | 0.9628 |
| MCC | 0.9628 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1610.4 (0.62 ms/张) |
| End-to-End FPS | 207.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9871 | 0.9845 | 0.9794 | 0.9786 | 0.9990 | 0.9988 | 195 |
| Malignant_lesions | 0.9367 | 0.9885 | 0.9487 | 0.9427 | 0.9857 | 0.9979 | 0.9913 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9891 | 0.9808 | 0.9903 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=3 TN=229 FN=5
- **Malignant_lesions**: TP=74 FP=4 TN=344 FN=5
- **body_of_stomach_normal**: TP=153 FP=3 TN=271 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 4 | 1 |
| Malignant_lesions | 3 | 74 | 2 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9845    0.9744    0.9794       195
     Malignant_lesions     0.9487    0.9367    0.9427        79
body_of_stomach_normal     0.9808    1.0000    0.9903       153

              accuracy                         0.9766       427
             macro avg     0.9713    0.9704    0.9708       427
          weighted avg     0.9765    0.9766    0.9765       427

```