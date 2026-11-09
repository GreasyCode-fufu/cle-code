# iformer_t 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/iformer_t_seed44/iformer_t/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9373 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9904 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9375 |
| Balanced Acc | 0.9357 |
| Macro-Sensitivity | 0.9357 |
| Macro-Specificity | 0.9682 |
| Macro-Precision | 0.9401 |
| AUC-PR (Macro) | 0.9834 |
| Cohen's Kappa | 0.9059 |
| MCC | 0.9065 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 521.61M |
| FLOPs | 1.04G |
| Params | 2.63M |
| Pure-Inference FPS | 6599.5 (0.15 ms/张) |
| End-to-End FPS | 245.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9402 | 0.9005 | 0.9277 | 0.9746 | 0.9909 | 0.9843 | 208 |
| Malignant_lesions | 0.8920 | 0.9800 | 0.9515 | 0.9208 | 0.9538 | 0.9833 | 0.9714 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9844 | 0.9684 | 0.9634 | 0.9793 | 0.9969 | 0.9943 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=22 TN=346 FN=9
- **Malignant_lesions**: TP=157 FP=8 TN=392 FN=19
- **body_of_stomach_normal**: TP=184 FP=6 TN=378 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 5 | 4 |
| Malignant_lesions | 17 | 157 | 2 |
| body_of_stomach_normal | 5 | 3 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9005    0.9567    0.9277       208
     Malignant_lesions     0.9515    0.8920    0.9208       176
body_of_stomach_normal     0.9684    0.9583    0.9634       192

              accuracy                         0.9375       576
             macro avg     0.9401    0.9357    0.9373       576
          weighted avg     0.9387    0.9375    0.9375       576

```