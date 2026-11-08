# starnet_s2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_seed42/starnet_s2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9450 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9920 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9447 |
| Balanced Acc | 0.9442 |
| Macro-Sensitivity | 0.9442 |
| Macro-Specificity | 0.9719 |
| Macro-Precision | 0.9462 |
| AUC-PR (Macro) | 0.9851 |
| Cohen's Kappa | 0.9164 |
| MCC | 0.9167 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.29M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3693.9 (0.27 ms/张) |
| End-to-End FPS | 142.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9484 | 0.9120 | 0.9292 | 0.9694 | 0.9864 | 0.9785 | 208 |
| Malignant_lesions | 0.9375 | 0.9750 | 0.9429 | 0.9402 | 0.9726 | 0.9920 | 0.9808 | 176 |
| body_of_stomach_normal | 0.9479 | 0.9922 | 0.9838 | 0.9655 | 0.9744 | 0.9977 | 0.9959 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=19 TN=349 FN=11
- **Malignant_lesions**: TP=165 FP=10 TN=390 FN=11
- **body_of_stomach_normal**: TP=182 FP=3 TN=381 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 9 | 165 | 2 |
| body_of_stomach_normal | 10 | 0 | 182 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9120    0.9471    0.9292       208
     Malignant_lesions     0.9429    0.9375    0.9402       176
body_of_stomach_normal     0.9838    0.9479    0.9655       192

              accuracy                         0.9444       576
             macro avg     0.9462    0.9442    0.9450       576
          weighted avg     0.9454    0.9444    0.9447       576

```