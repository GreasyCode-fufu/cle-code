# convnext_wtb_msf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wtb_msf_seed42/convnext_wtb_msf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9726 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9982 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9786 |
| Balanced Acc | 0.9645 |
| Macro-Sensitivity | 0.9645 |
| Macro-Specificity | 0.9875 |
| Macro-Precision | 0.9822 |
| AUC-PR (Macro) | 0.9949 |
| Cohen's Kappa | 0.9662 |
| MCC | 0.9667 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.52G |
| FLOPs | 9.05G |
| Params | 29.54M |
| Pure-Inference FPS | 1144.3 (0.87 ms/张) |
| End-to-End FPS | 247.3 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9949 | 0.9655 | 0.9604 | 0.9773 | 0.9956 | 0.9978 | 0.9973 | 195 |
| Malignant_lesions | 0.8987 | 0.9971 | 0.9861 | 0.9404 | 0.9775 | 0.9967 | 0.9874 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=8 TN=224 FN=1
- **Malignant_lesions**: TP=71 FP=1 TN=347 FN=8
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 1 | 0 |
| Malignant_lesions | 8 | 71 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9604    0.9949    0.9773       195
     Malignant_lesions     0.9861    0.8987    0.9404        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9789       427
             macro avg     0.9822    0.9645    0.9726       427
          weighted avg     0.9793    0.9789    0.9786       427

```