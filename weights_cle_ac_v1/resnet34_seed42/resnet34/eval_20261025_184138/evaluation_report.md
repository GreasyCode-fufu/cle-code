# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/resnet34_seed42/resnet34/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9289 |
| Accuracy | 0.9288 |
| Macro-AUROC (OvR) | 0.9917 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9291 |
| Balanced Acc | 0.9252 |
| Macro-Sensitivity | 0.9252 |
| Macro-Specificity | 0.9633 |
| Macro-Precision | 0.9383 |
| AUC-PR (Macro) | 0.9837 |
| Cohen's Kappa | 0.8926 |
| MCC | 0.8955 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 1304.9 (0.77 ms/张) |
| End-to-End FPS | 244.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9049 | 0.8529 | 0.9103 | 0.9852 | 0.9881 | 0.9813 | 208 |
| Malignant_lesions | 0.8466 | 0.9875 | 0.9675 | 0.9030 | 0.9360 | 0.9882 | 0.9721 | 176 |
| body_of_stomach_normal | 0.9531 | 0.9974 | 0.9946 | 0.9734 | 0.9770 | 0.9988 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=35 TN=333 FN=5
- **Malignant_lesions**: TP=149 FP=5 TN=395 FN=27
- **body_of_stomach_normal**: TP=183 FP=1 TN=383 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 5 | 0 |
| Malignant_lesions | 26 | 149 | 1 |
| body_of_stomach_normal | 9 | 0 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8529    0.9760    0.9103       208
     Malignant_lesions     0.9675    0.8466    0.9030       176
body_of_stomach_normal     0.9946    0.9531    0.9734       192

              accuracy                         0.9288       576
             macro avg     0.9383    0.9252    0.9289       576
          weighted avg     0.9352    0.9288    0.9291       576

```