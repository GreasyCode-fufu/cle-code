# convnext_v6x3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v6x3_seed42/convnext_v6x3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9529 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9954 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9531 |
| Balanced Acc | 0.9548 |
| Macro-Sensitivity | 0.9548 |
| Macro-Specificity | 0.9771 |
| Macro-Precision | 0.9527 |
| AUC-PR (Macro) | 0.9919 |
| Cohen's Kappa | 0.9297 |
| MCC | 0.9306 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.69M |
| Pure-Inference FPS | 955.4 (1.05 ms/张) |
| End-to-End FPS | 228.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9038 | 0.9891 | 0.9792 | 0.9400 | 0.9479 | 0.9927 | 0.9897 | 208 |
| Malignant_lesions | 0.9659 | 0.9525 | 0.8995 | 0.9315 | 0.9845 | 0.9948 | 0.9890 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9896 | 0.9795 | 0.9871 | 0.9974 | 0.9987 | 0.9971 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=4 TN=364 FN=20
- **Malignant_lesions**: TP=170 FP=19 TN=381 FN=6
- **body_of_stomach_normal**: TP=191 FP=4 TN=380 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 19 | 1 |
| Malignant_lesions | 3 | 170 | 3 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9792    0.9038    0.9400       208
     Malignant_lesions     0.8995    0.9659    0.9315       176
body_of_stomach_normal     0.9795    0.9948    0.9871       192

              accuracy                         0.9531       576
             macro avg     0.9527    0.9548    0.9529       576
          weighted avg     0.9549    0.9531    0.9531       576

```