# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2m_seed42/convnext_v2m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9560 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9959 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9564 |
| Balanced Acc | 0.9560 |
| Macro-Sensitivity | 0.9560 |
| Macro-Specificity | 0.9783 |
| Macro-Precision | 0.9562 |
| AUC-PR (Macro) | 0.9927 |
| Cohen's Kappa | 0.9347 |
| MCC | 0.9349 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 471.3 (2.12 ms/张) |
| End-to-End FPS | 187.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9755 | 0.9563 | 0.9517 | 0.9703 | 0.9939 | 0.9918 | 208 |
| Malignant_lesions | 0.9261 | 0.9775 | 0.9477 | 0.9368 | 0.9678 | 0.9947 | 0.9880 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9818 | 0.9646 | 0.9795 | 0.9974 | 0.9993 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=9 TN=359 FN=11
- **Malignant_lesions**: TP=163 FP=9 TN=391 FN=13
- **body_of_stomach_normal**: TP=191 FP=7 TN=377 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 8 | 163 | 5 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9563    0.9471    0.9517       208
     Malignant_lesions     0.9477    0.9261    0.9368       176
body_of_stomach_normal     0.9646    0.9948    0.9795       192

              accuracy                         0.9566       576
             macro avg     0.9562    0.9560    0.9560       576
          weighted avg     0.9565    0.9566    0.9564       576

```