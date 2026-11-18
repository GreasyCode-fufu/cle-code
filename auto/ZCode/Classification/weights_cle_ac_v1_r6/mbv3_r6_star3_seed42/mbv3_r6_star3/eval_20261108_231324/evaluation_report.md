# mbv3_r6_star3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star3_seed42/mbv3_r6_star3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9530 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9918 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9531 |
| Balanced Acc | 0.9514 |
| Macro-Sensitivity | 0.9514 |
| Macro-Specificity | 0.9760 |
| Macro-Precision | 0.9558 |
| AUC-PR (Macro) | 0.9846 |
| Cohen's Kappa | 0.9294 |
| MCC | 0.9300 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.97M |
| FLOPs | 463.94M |
| Params | 3.26M |
| Pure-Inference FPS | 5989.2 (0.17 ms/张) |
| End-to-End FPS | 305.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9484 | 0.9136 | 0.9393 | 0.9803 | 0.9879 | 0.9749 | 208 |
| Malignant_lesions | 0.9034 | 0.9875 | 0.9695 | 0.9353 | 0.9587 | 0.9878 | 0.9793 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9922 | 0.9844 | 0.9844 | 0.9922 | 0.9998 | 0.9996 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=19 TN=349 FN=7
- **Malignant_lesions**: TP=159 FP=5 TN=395 FN=17
- **body_of_stomach_normal**: TP=189 FP=3 TN=381 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 16 | 159 | 1 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9136    0.9663    0.9393       208
     Malignant_lesions     0.9695    0.9034    0.9353       176
body_of_stomach_normal     0.9844    0.9844    0.9844       192

              accuracy                         0.9531       576
             macro avg     0.9558    0.9514    0.9530       576
          weighted avg     0.9543    0.9531    0.9531       576

```