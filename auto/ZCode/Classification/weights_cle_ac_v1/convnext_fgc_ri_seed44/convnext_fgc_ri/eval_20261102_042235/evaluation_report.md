# convnext_fgc_ri 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_ri_seed44/convnext_fgc_ri/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9636 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9636 |
| Balanced Acc | 0.9633 |
| Macro-Sensitivity | 0.9633 |
| Macro-Specificity | 0.9816 |
| Macro-Precision | 0.9638 |
| AUC-PR (Macro) | 0.9902 |
| Cohen's Kappa | 0.9452 |
| MCC | 0.9452 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 1511.8 (0.66 ms/张) |
| End-to-End FPS | 262.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9701 | 0.9476 | 0.9522 | 0.9754 | 0.9922 | 0.9890 | 208 |
| Malignant_lesions | 0.9489 | 0.9800 | 0.9543 | 0.9516 | 0.9776 | 0.9926 | 0.9820 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9948 | 0.9895 | 0.9869 | 0.9922 | 0.9998 | 0.9996 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=11 TN=357 FN=9
- **Malignant_lesions**: TP=167 FP=8 TN=392 FN=9
- **body_of_stomach_normal**: TP=189 FP=2 TN=382 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 8 | 167 | 1 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9476    0.9567    0.9522       208
     Malignant_lesions     0.9543    0.9489    0.9516       176
body_of_stomach_normal     0.9895    0.9844    0.9869       192

              accuracy                         0.9635       576
             macro avg     0.9638    0.9633    0.9636       576
          weighted avg     0.9636    0.9635    0.9636       576

```