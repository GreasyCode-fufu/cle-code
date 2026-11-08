# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz8_dce4m_seed44/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9704 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9972 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9766 |
| Balanced Acc | 0.9704 |
| Macro-Sensitivity | 0.9704 |
| Macro-Specificity | 0.9880 |
| Macro-Precision | 0.9704 |
| AUC-PR (Macro) | 0.9916 |
| Cohen's Kappa | 0.9628 |
| MCC | 0.9628 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3154.9 (0.32 ms/张) |
| End-to-End FPS | 18.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9784 | 0.9744 | 0.9744 | 0.9784 | 0.9969 | 0.9964 | 195 |
| Malignant_lesions | 0.9367 | 0.9856 | 0.9367 | 0.9367 | 0.9856 | 0.9945 | 0.9784 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=5 TN=227 FN=5
- **Malignant_lesions**: TP=74 FP=5 TN=343 FN=5
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 5 | 0 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9744    0.9744    0.9744       195
     Malignant_lesions     0.9367    0.9367    0.9367        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9766       427
             macro avg     0.9704    0.9704    0.9704       427
          weighted avg     0.9766    0.9766    0.9766       427

```