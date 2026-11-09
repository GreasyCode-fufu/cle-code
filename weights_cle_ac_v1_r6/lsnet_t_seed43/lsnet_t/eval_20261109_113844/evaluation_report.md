# lsnet_t 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/lsnet_t_seed43/lsnet_t/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9136 |
| Accuracy | 0.9149 |
| Macro-AUROC (OvR) | 0.9891 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9140 |
| Balanced Acc | 0.9120 |
| Macro-Sensitivity | 0.9120 |
| Macro-Specificity | 0.9567 |
| Macro-Precision | 0.9203 |
| AUC-PR (Macro) | 0.9803 |
| Cohen's Kappa | 0.8718 |
| MCC | 0.8738 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 320.32M |
| FLOPs | 640.63M |
| Params | 11.02M |
| Pure-Inference FPS | 5204.7 (0.19 ms/张) |
| End-to-End FPS | 248.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9293 | 0.8813 | 0.9040 | 0.9580 | 0.9815 | 0.9712 | 208 |
| Malignant_lesions | 0.8239 | 0.9875 | 0.9667 | 0.8896 | 0.9272 | 0.9902 | 0.9778 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9531 | 0.9130 | 0.9474 | 0.9919 | 0.9957 | 0.9918 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=26 TN=342 FN=15
- **Malignant_lesions**: TP=145 FP=5 TN=395 FN=31
- **body_of_stomach_normal**: TP=189 FP=18 TN=366 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 5 | 10 |
| Malignant_lesions | 23 | 145 | 8 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8813    0.9279    0.9040       208
     Malignant_lesions     0.9667    0.8239    0.8896       176
body_of_stomach_normal     0.9130    0.9844    0.9474       192

              accuracy                         0.9149       576
             macro avg     0.9203    0.9120    0.9136       576
          weighted avg     0.9180    0.9149    0.9140       576

```