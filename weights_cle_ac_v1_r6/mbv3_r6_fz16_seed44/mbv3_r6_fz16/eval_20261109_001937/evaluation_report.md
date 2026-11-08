# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz16_seed44/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9530 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9954 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9530 |
| Balanced Acc | 0.9524 |
| Macro-Sensitivity | 0.9524 |
| Macro-Specificity | 0.9763 |
| Macro-Precision | 0.9538 |
| AUC-PR (Macro) | 0.9910 |
| Cohen's Kappa | 0.9295 |
| MCC | 0.9296 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 5067.1 (0.20 ms/张) |
| End-to-End FPS | 184.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9647 | 0.9384 | 0.9451 | 0.9726 | 0.9934 | 0.9897 | 208 |
| Malignant_lesions | 0.9261 | 0.9825 | 0.9588 | 0.9422 | 0.9680 | 0.9951 | 0.9867 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9818 | 0.9641 | 0.9716 | 0.9895 | 0.9978 | 0.9966 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=13 TN=355 FN=10
- **Malignant_lesions**: TP=163 FP=7 TN=393 FN=13
- **body_of_stomach_normal**: TP=188 FP=7 TN=377 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 9 | 163 | 4 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9384    0.9519    0.9451       208
     Malignant_lesions     0.9588    0.9261    0.9422       176
body_of_stomach_normal     0.9641    0.9792    0.9716       192

              accuracy                         0.9531       576
             macro avg     0.9538    0.9524    0.9530       576
          weighted avg     0.9532    0.9531    0.9530       576

```