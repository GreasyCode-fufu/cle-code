# mbv3_sft_head 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_head_seed42/mbv3_sft_head/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8646 |
| Accuracy | 0.8663 |
| Macro-AUROC (OvR) | 0.9745 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8645 |
| Balanced Acc | 0.8657 |
| Macro-Sensitivity | 0.8657 |
| Macro-Specificity | 0.9327 |
| Macro-Precision | 0.8691 |
| AUC-PR (Macro) | 0.9541 |
| Cohen's Kappa | 0.7989 |
| MCC | 0.8014 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3602.5 (0.28 ms/张) |
| End-to-End FPS | 33.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8173 | 0.9239 | 0.8586 | 0.8374 | 0.8995 | 0.9568 | 0.9314 | 208 |
| Malignant_lesions | 0.7955 | 0.9600 | 0.8974 | 0.8434 | 0.9143 | 0.9754 | 0.9477 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9141 | 0.8514 | 0.9130 | 0.9915 | 0.9913 | 0.9831 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=170 FP=28 TN=340 FN=38
- **Malignant_lesions**: TP=140 FP=16 TN=384 FN=36
- **body_of_stomach_normal**: TP=189 FP=33 TN=351 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 170 | 15 | 23 |
| Malignant_lesions | 26 | 140 | 10 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8586    0.8173    0.8374       208
     Malignant_lesions     0.8974    0.7955    0.8434       176
body_of_stomach_normal     0.8514    0.9844    0.9130       192

              accuracy                         0.8663       576
             macro avg     0.8691    0.8657    0.8646       576
          weighted avg     0.8680    0.8663    0.8645       576

```