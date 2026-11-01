# convnext_fgc4_dsd1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc4_dsd1_seed42/convnext_fgc4_dsd1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9473 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9930 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9477 |
| Balanced Acc | 0.9463 |
| Macro-Sensitivity | 0.9463 |
| Macro-Specificity | 0.9736 |
| Macro-Precision | 0.9491 |
| AUC-PR (Macro) | 0.9873 |
| Cohen's Kappa | 0.9216 |
| MCC | 0.9219 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.42G |
| FLOPs | 8.84G |
| Params | 27.68M |
| Pure-Inference FPS | 1011.7 (0.99 ms/张) |
| End-to-End FPS | 234.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9538 | 0.9213 | 0.9387 | 0.9750 | 0.9917 | 0.9857 | 208 |
| Malignant_lesions | 0.8977 | 0.9800 | 0.9518 | 0.9240 | 0.9561 | 0.9886 | 0.9789 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9870 | 0.9742 | 0.9793 | 0.9921 | 0.9987 | 0.9973 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=17 TN=351 FN=9
- **Malignant_lesions**: TP=158 FP=8 TN=392 FN=18
- **body_of_stomach_normal**: TP=189 FP=5 TN=379 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 15 | 158 | 3 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9213    0.9567    0.9387       208
     Malignant_lesions     0.9518    0.8977    0.9240       176
body_of_stomach_normal     0.9742    0.9844    0.9793       192

              accuracy                         0.9479       576
             macro avg     0.9491    0.9463    0.9473       576
          weighted avg     0.9483    0.9479    0.9477       576

```