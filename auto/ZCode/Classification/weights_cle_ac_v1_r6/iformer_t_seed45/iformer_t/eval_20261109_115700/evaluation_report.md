# iformer_t 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/iformer_t_seed45/iformer_t/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9472 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9923 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9476 |
| Balanced Acc | 0.9451 |
| Macro-Sensitivity | 0.9451 |
| Macro-Specificity | 0.9733 |
| Macro-Precision | 0.9521 |
| AUC-PR (Macro) | 0.9844 |
| Cohen's Kappa | 0.9215 |
| MCC | 0.9228 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 521.61M |
| FLOPs | 1.04G |
| Params | 2.63M |
| Pure-Inference FPS | 5288.3 (0.19 ms/张) |
| End-to-End FPS | 173.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9402 | 0.9022 | 0.9376 | 0.9858 | 0.9907 | 0.9732 | 208 |
| Malignant_lesions | 0.8750 | 0.9900 | 0.9747 | 0.9222 | 0.9474 | 0.9866 | 0.9810 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9896 | 0.9793 | 0.9818 | 0.9922 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=22 TN=346 FN=5
- **Malignant_lesions**: TP=154 FP=4 TN=396 FN=22
- **body_of_stomach_normal**: TP=189 FP=4 TN=380 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 4 | 1 |
| Malignant_lesions | 19 | 154 | 3 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9022    0.9760    0.9376       208
     Malignant_lesions     0.9747    0.8750    0.9222       176
body_of_stomach_normal     0.9793    0.9844    0.9818       192

              accuracy                         0.9479       576
             macro avg     0.9521    0.9451    0.9472       576
          weighted avg     0.9500    0.9479    0.9476       576

```