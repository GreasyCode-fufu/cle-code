# convnext_tiny_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_dce4m_seed44/convnext_tiny_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9629 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9973 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9633 |
| Balanced Acc | 0.9617 |
| Macro-Sensitivity | 0.9617 |
| Macro-Specificity | 0.9814 |
| Macro-Precision | 0.9652 |
| AUC-PR (Macro) | 0.9947 |
| Cohen's Kappa | 0.9451 |
| MCC | 0.9456 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1659.5 (0.60 ms/张) |
| End-to-End FPS | 8.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9674 | 0.9444 | 0.9623 | 0.9889 | 0.9961 | 0.9936 | 208 |
| Malignant_lesions | 0.9148 | 0.9925 | 0.9817 | 0.9471 | 0.9636 | 0.9965 | 0.9919 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9844 | 0.9694 | 0.9794 | 0.9947 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=12 TN=356 FN=4
- **Malignant_lesions**: TP=161 FP=3 TN=397 FN=15
- **body_of_stomach_normal**: TP=190 FP=6 TN=378 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 3 | 1 |
| Malignant_lesions | 10 | 161 | 5 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9444    0.9808    0.9623       208
     Malignant_lesions     0.9817    0.9148    0.9471       176
body_of_stomach_normal     0.9694    0.9896    0.9794       192

              accuracy                         0.9635       576
             macro avg     0.9652    0.9617    0.9629       576
          weighted avg     0.9641    0.9635    0.9633       576

```