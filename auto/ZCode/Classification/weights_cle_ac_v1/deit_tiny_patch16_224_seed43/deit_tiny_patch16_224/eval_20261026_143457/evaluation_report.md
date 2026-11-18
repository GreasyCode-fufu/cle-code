# deit_tiny_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/deit_tiny_patch16_224_seed43/deit_tiny_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9697 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9975 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9763 |
| Balanced Acc | 0.9628 |
| Macro-Sensitivity | 0.9628 |
| Macro-Specificity | 0.9866 |
| Macro-Precision | 0.9776 |
| AUC-PR (Macro) | 0.9927 |
| Cohen's Kappa | 0.9625 |
| MCC | 0.9629 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 5.49M |
| Pure-Inference FPS | 5218.6 (0.19 ms/张) |
| End-to-End FPS | 262.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9897 | 0.9655 | 0.9602 | 0.9747 | 0.9912 | 0.9972 | 0.9967 | 195 |
| Malignant_lesions | 0.8987 | 0.9943 | 0.9726 | 0.9342 | 0.9774 | 0.9953 | 0.9814 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=8 TN=224 FN=2
- **Malignant_lesions**: TP=71 FP=2 TN=346 FN=8
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 2 | 0 |
| Malignant_lesions | 8 | 71 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9602    0.9897    0.9747       195
     Malignant_lesions     0.9726    0.8987    0.9342        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9766       427
             macro avg     0.9776    0.9628    0.9697       427
          weighted avg     0.9768    0.9766    0.9763       427

```