# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/pvt_tiny_seed44/pvt_tiny/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8999 |
| Accuracy | 0.9045 |
| Macro-AUROC (OvR) | 0.9920 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9016 |
| Balanced Acc | 0.8979 |
| Macro-Sensitivity | 0.8979 |
| Macro-Specificity | 0.9509 |
| Macro-Precision | 0.9166 |
| AUC-PR (Macro) | 0.9844 |
| Cohen's Kappa | 0.8556 |
| MCC | 0.8618 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 643.9 (1.55 ms/张) |
| End-to-End FPS | 206.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.8913 | 0.8347 | 0.8978 | 0.9820 | 0.9889 | 0.9820 | 208 |
| Malignant_lesions | 0.7330 | 0.9900 | 0.9699 | 0.8350 | 0.8939 | 0.9882 | 0.9734 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9714 | 0.9453 | 0.9669 | 0.9947 | 0.9988 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=40 TN=328 FN=6
- **Malignant_lesions**: TP=129 FP=4 TN=396 FN=47
- **body_of_stomach_normal**: TP=190 FP=11 TN=373 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 38 | 129 | 9 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8347    0.9712    0.8978       208
     Malignant_lesions     0.9699    0.7330    0.8350       176
body_of_stomach_normal     0.9453    0.9896    0.9669       192

              accuracy                         0.9045       576
             macro avg     0.9166    0.8979    0.8999       576
          weighted avg     0.9129    0.9045    0.9016       576

```