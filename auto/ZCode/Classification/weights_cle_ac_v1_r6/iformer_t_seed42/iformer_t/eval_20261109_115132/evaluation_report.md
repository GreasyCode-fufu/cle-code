# iformer_t 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/iformer_t_seed42/iformer_t/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9527 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9937 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9528 |
| Balanced Acc | 0.9515 |
| Macro-Sensitivity | 0.9515 |
| Macro-Specificity | 0.9761 |
| Macro-Precision | 0.9554 |
| AUC-PR (Macro) | 0.9887 |
| Cohen's Kappa | 0.9294 |
| MCC | 0.9300 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 521.61M |
| FLOPs | 1.04G |
| Params | 2.63M |
| Pure-Inference FPS | 4942.9 (0.20 ms/张) |
| End-to-End FPS | 249.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9538 | 0.9213 | 0.9387 | 0.9750 | 0.9892 | 0.9842 | 208 |
| Malignant_lesions | 0.8977 | 0.9900 | 0.9753 | 0.9349 | 0.9565 | 0.9932 | 0.9849 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9844 | 0.9697 | 0.9846 | 1.0000 | 0.9987 | 0.9970 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=17 TN=351 FN=9
- **Malignant_lesions**: TP=158 FP=4 TN=396 FN=18
- **body_of_stomach_normal**: TP=192 FP=6 TN=378 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 4 | 5 |
| Malignant_lesions | 17 | 158 | 1 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9213    0.9567    0.9387       208
     Malignant_lesions     0.9753    0.8977    0.9349       176
body_of_stomach_normal     0.9697    1.0000    0.9846       192

              accuracy                         0.9531       576
             macro avg     0.9554    0.9515    0.9527       576
          weighted avg     0.9539    0.9531    0.9528       576

```