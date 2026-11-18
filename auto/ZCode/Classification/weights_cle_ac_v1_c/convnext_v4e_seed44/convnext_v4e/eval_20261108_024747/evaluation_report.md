# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4e_seed44/convnext_v4e/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9527 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9957 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9530 |
| Balanced Acc | 0.9514 |
| Macro-Sensitivity | 0.9514 |
| Macro-Specificity | 0.9761 |
| Macro-Precision | 0.9550 |
| AUC-PR (Macro) | 0.9915 |
| Cohen's Kappa | 0.9294 |
| MCC | 0.9299 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 535.9 (1.87 ms/张) |
| End-to-End FPS | 197.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9538 | 0.9220 | 0.9437 | 0.9804 | 0.9935 | 0.9900 | 208 |
| Malignant_lesions | 0.9034 | 0.9850 | 0.9636 | 0.9326 | 0.9586 | 0.9945 | 0.9865 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9896 | 0.9793 | 0.9818 | 0.9922 | 0.9990 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=17 TN=351 FN=7
- **Malignant_lesions**: TP=159 FP=6 TN=394 FN=17
- **body_of_stomach_normal**: TP=189 FP=4 TN=380 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 14 | 159 | 3 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9220    0.9663    0.9437       208
     Malignant_lesions     0.9636    0.9034    0.9326       176
body_of_stomach_normal     0.9793    0.9844    0.9818       192

              accuracy                         0.9531       576
             macro avg     0.9550    0.9514    0.9527       576
          weighted avg     0.9538    0.9531    0.9530       576

```