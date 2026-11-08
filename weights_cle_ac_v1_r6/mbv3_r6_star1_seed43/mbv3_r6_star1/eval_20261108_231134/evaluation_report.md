# mbv3_r6_star1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star1_seed43/mbv3_r6_star1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9391 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9918 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9392 |
| Balanced Acc | 0.9387 |
| Macro-Sensitivity | 0.9387 |
| Macro-Specificity | 0.9694 |
| Macro-Precision | 0.9397 |
| AUC-PR (Macro) | 0.9853 |
| Cohen's Kappa | 0.9086 |
| MCC | 0.9087 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.18M |
| FLOPs | 464.35M |
| Params | 3.47M |
| Pure-Inference FPS | 4225.2 (0.24 ms/张) |
| End-to-End FPS | 257.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9511 | 0.9147 | 0.9212 | 0.9589 | 0.9877 | 0.9805 | 208 |
| Malignant_lesions | 0.9091 | 0.9700 | 0.9302 | 0.9195 | 0.9604 | 0.9890 | 0.9777 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9870 | 0.9741 | 0.9766 | 0.9896 | 0.9988 | 0.9976 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=18 TN=350 FN=15
- **Malignant_lesions**: TP=160 FP=12 TN=388 FN=16
- **body_of_stomach_normal**: TP=188 FP=5 TN=379 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 12 | 3 |
| Malignant_lesions | 14 | 160 | 2 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9147    0.9279    0.9212       208
     Malignant_lesions     0.9302    0.9091    0.9195       176
body_of_stomach_normal     0.9741    0.9792    0.9766       192

              accuracy                         0.9392       576
             macro avg     0.9397    0.9387    0.9391       576
          weighted avg     0.9392    0.9392    0.9392       576

```