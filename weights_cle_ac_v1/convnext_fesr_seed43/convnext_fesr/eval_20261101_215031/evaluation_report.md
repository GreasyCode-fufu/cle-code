# convnext_fesr 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fesr_seed43/convnext_fesr/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9525 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9958 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9530 |
| Balanced Acc | 0.9523 |
| Macro-Sensitivity | 0.9523 |
| Macro-Specificity | 0.9765 |
| Macro-Precision | 0.9534 |
| AUC-PR (Macro) | 0.9922 |
| Cohen's Kappa | 0.9295 |
| MCC | 0.9297 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 663.4 (1.51 ms/张) |
| End-to-End FPS | 200.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9755 | 0.9565 | 0.9542 | 0.9729 | 0.9951 | 0.9914 | 208 |
| Malignant_lesions | 0.9205 | 0.9825 | 0.9586 | 0.9391 | 0.9656 | 0.9935 | 0.9875 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9714 | 0.9450 | 0.9643 | 0.9920 | 0.9988 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=9 TN=359 FN=10
- **Malignant_lesions**: TP=162 FP=7 TN=393 FN=14
- **body_of_stomach_normal**: TP=189 FP=11 TN=373 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 6 | 162 | 8 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9565    0.9519    0.9542       208
     Malignant_lesions     0.9586    0.9205    0.9391       176
body_of_stomach_normal     0.9450    0.9844    0.9643       192

              accuracy                         0.9531       576
             macro avg     0.9534    0.9523    0.9525       576
          weighted avg     0.9533    0.9531    0.9530       576

```