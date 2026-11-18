# starnet_s2_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_seed45/starnet_s2_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9228 |
| Accuracy | 0.9236 |
| Macro-AUROC (OvR) | 0.9882 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9231 |
| Balanced Acc | 0.9212 |
| Macro-Sensitivity | 0.9212 |
| Macro-Specificity | 0.9611 |
| Macro-Precision | 0.9268 |
| AUC-PR (Macro) | 0.9781 |
| Cohen's Kappa | 0.8849 |
| MCC | 0.8859 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.29M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3407.5 (0.29 ms/张) |
| End-to-End FPS | 128.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9266 | 0.8778 | 0.9044 | 0.9606 | 0.9823 | 0.9722 | 208 |
| Malignant_lesions | 0.8466 | 0.9775 | 0.9430 | 0.8922 | 0.9354 | 0.9835 | 0.9642 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9792 | 0.9594 | 0.9717 | 0.9921 | 0.9988 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=27 TN=341 FN=14
- **Malignant_lesions**: TP=149 FP=9 TN=391 FN=27
- **body_of_stomach_normal**: TP=189 FP=8 TN=376 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 9 | 5 |
| Malignant_lesions | 24 | 149 | 3 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8778    0.9327    0.9044       208
     Malignant_lesions     0.9430    0.8466    0.8922       176
body_of_stomach_normal     0.9594    0.9844    0.9717       192

              accuracy                         0.9236       576
             macro avg     0.9268    0.9212    0.9228       576
          weighted avg     0.9249    0.9236    0.9231       576

```