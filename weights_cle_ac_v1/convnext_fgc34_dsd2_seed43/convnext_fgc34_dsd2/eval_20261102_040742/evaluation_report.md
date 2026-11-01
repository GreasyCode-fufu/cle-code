# convnext_fgc34_dsd2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc34_dsd2_seed43/convnext_fgc34_dsd2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9802 |
| Accuracy | 0.9836 |
| Macro-AUROC (OvR) | 0.9979 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9837 |
| Balanced Acc | 0.9825 |
| Macro-Sensitivity | 0.9825 |
| Macro-Specificity | 0.9921 |
| Macro-Precision | 0.9779 |
| AUC-PR (Macro) | 0.9939 |
| Cohen's Kappa | 0.9740 |
| MCC | 0.9740 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.40G |
| FLOPs | 8.80G |
| Params | 27.49M |
| Pure-Inference FPS | 1686.1 (0.59 ms/张) |
| End-to-End FPS | 256.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9914 | 0.9896 | 0.9845 | 0.9829 | 0.9976 | 0.9972 | 195 |
| Malignant_lesions | 0.9747 | 0.9885 | 0.9506 | 0.9625 | 0.9942 | 0.9961 | 0.9845 | 79 |
| body_of_stomach_normal | 0.9935 | 0.9964 | 0.9935 | 0.9935 | 0.9964 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=2 TN=230 FN=4
- **Malignant_lesions**: TP=77 FP=4 TN=344 FN=2
- **body_of_stomach_normal**: TP=152 FP=1 TN=273 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 4 | 0 |
| Malignant_lesions | 1 | 77 | 1 |
| body_of_stomach_normal | 1 | 0 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9896    0.9795    0.9845       195
     Malignant_lesions     0.9506    0.9747    0.9625        79
body_of_stomach_normal     0.9935    0.9935    0.9935       153

              accuracy                         0.9836       427
             macro avg     0.9779    0.9825    0.9802       427
          weighted avg     0.9838    0.9836    0.9837       427

```