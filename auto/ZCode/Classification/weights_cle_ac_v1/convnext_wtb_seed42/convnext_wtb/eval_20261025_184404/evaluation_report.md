# convnext_wtb 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wtb_seed42/convnext_wtb/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9648 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9651 |
| Balanced Acc | 0.9634 |
| Macro-Sensitivity | 0.9634 |
| Macro-Specificity | 0.9822 |
| Macro-Precision | 0.9673 |
| AUC-PR (Macro) | 0.9929 |
| Cohen's Kappa | 0.9477 |
| MCC | 0.9482 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.50G |
| FLOPs | 9.00G |
| Params | 29.10M |
| Pure-Inference FPS | 779.3 (1.28 ms/张) |
| End-to-End FPS | 220.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9620 | 0.9358 | 0.9577 | 0.9888 | 0.9948 | 0.9908 | 208 |
| Malignant_lesions | 0.9148 | 0.9925 | 0.9817 | 0.9471 | 0.9636 | 0.9946 | 0.9889 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9922 | 0.9845 | 0.9896 | 0.9974 | 0.9995 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=14 TN=354 FN=4
- **Malignant_lesions**: TP=161 FP=3 TN=397 FN=15
- **body_of_stomach_normal**: TP=191 FP=3 TN=381 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 3 | 1 |
| Malignant_lesions | 13 | 161 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9358    0.9808    0.9577       208
     Malignant_lesions     0.9817    0.9148    0.9471       176
body_of_stomach_normal     0.9845    0.9948    0.9896       192

              accuracy                         0.9653       576
             macro avg     0.9673    0.9634    0.9648       576
          weighted avg     0.9661    0.9653    0.9651       576

```