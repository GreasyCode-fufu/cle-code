# convnext_v4d 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4d_seed43/convnext_v4d/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9264 |
| Accuracy | 0.9271 |
| Macro-AUROC (OvR) | 0.9938 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9268 |
| Balanced Acc | 0.9229 |
| Macro-Sensitivity | 0.9229 |
| Macro-Specificity | 0.9623 |
| Macro-Precision | 0.9372 |
| AUC-PR (Macro) | 0.9891 |
| Cohen's Kappa | 0.8899 |
| MCC | 0.8934 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.59G |
| FLOPs | 7.18G |
| Params | 22.94M |
| Pure-Inference FPS | 390.5 (2.56 ms/张) |
| End-to-End FPS | 174.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9049 | 0.8529 | 0.9103 | 0.9852 | 0.9916 | 0.9851 | 208 |
| Malignant_lesions | 0.8239 | 0.9925 | 0.9797 | 0.8951 | 0.9276 | 0.9904 | 0.9834 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9896 | 0.9789 | 0.9738 | 0.9845 | 0.9993 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=35 TN=333 FN=5
- **Malignant_lesions**: TP=145 FP=3 TN=397 FN=31
- **body_of_stomach_normal**: TP=186 FP=4 TN=380 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 3 | 2 |
| Malignant_lesions | 29 | 145 | 2 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8529    0.9760    0.9103       208
     Malignant_lesions     0.9797    0.8239    0.8951       176
body_of_stomach_normal     0.9789    0.9688    0.9738       192

              accuracy                         0.9271       576
             macro avg     0.9372    0.9229    0.9264       576
          weighted avg     0.9337    0.9271    0.9268       576

```