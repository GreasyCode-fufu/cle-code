# starnet_s2_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce4m_seed43/starnet_s2_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9733 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9982 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9786 |
| Balanced Acc | 0.9670 |
| Macro-Sensitivity | 0.9670 |
| Macro-Specificity | 0.9882 |
| Macro-Precision | 0.9806 |
| AUC-PR (Macro) | 0.9950 |
| Cohen's Kappa | 0.9663 |
| MCC | 0.9666 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3780.4 (0.26 ms/张) |
| End-to-End FPS | 7.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9897 | 0.9784 | 0.9747 | 0.9822 | 0.9913 | 0.9982 | 0.9978 | 195 |
| Malignant_lesions | 0.9114 | 0.9971 | 0.9863 | 0.9474 | 0.9802 | 0.9964 | 0.9872 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9891 | 0.9808 | 0.9903 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=5 TN=227 FN=2
- **Malignant_lesions**: TP=72 FP=1 TN=347 FN=7
- **body_of_stomach_normal**: TP=153 FP=3 TN=271 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 1 | 1 |
| Malignant_lesions | 5 | 72 | 2 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9747    0.9897    0.9822       195
     Malignant_lesions     0.9863    0.9114    0.9474        79
body_of_stomach_normal     0.9808    1.0000    0.9903       153

              accuracy                         0.9789       427
             macro avg     0.9806    0.9670    0.9733       427
          weighted avg     0.9790    0.9789    0.9786       427

```