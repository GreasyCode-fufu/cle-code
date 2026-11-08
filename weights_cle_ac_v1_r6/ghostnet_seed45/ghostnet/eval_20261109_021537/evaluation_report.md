# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/ghostnet_seed45/ghostnet/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9373 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9916 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9372 |
| Balanced Acc | 0.9364 |
| Macro-Sensitivity | 0.9364 |
| Macro-Specificity | 0.9683 |
| Macro-Precision | 0.9393 |
| AUC-PR (Macro) | 0.9841 |
| Cohen's Kappa | 0.9059 |
| MCC | 0.9064 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 2891.4 (0.35 ms/张) |
| End-to-End FPS | 211.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9484 | 0.9108 | 0.9216 | 0.9614 | 0.9869 | 0.9773 | 208 |
| Malignant_lesions | 0.8920 | 0.9825 | 0.9573 | 0.9235 | 0.9539 | 0.9893 | 0.9775 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9740 | 0.9497 | 0.9668 | 0.9920 | 0.9987 | 0.9974 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=19 TN=349 FN=14
- **Malignant_lesions**: TP=157 FP=7 TN=393 FN=19
- **body_of_stomach_normal**: TP=189 FP=10 TN=374 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 7 | 7 |
| Malignant_lesions | 16 | 157 | 3 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9108    0.9327    0.9216       208
     Malignant_lesions     0.9573    0.8920    0.9235       176
body_of_stomach_normal     0.9497    0.9844    0.9668       192

              accuracy                         0.9375       576
             macro avg     0.9393    0.9364    0.9373       576
          weighted avg     0.9380    0.9375    0.9372       576

```