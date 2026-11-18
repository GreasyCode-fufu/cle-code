# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-30/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8906 |
| Accuracy | 0.8897 |
| Macro-AUROC (OvR) | 0.9757 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8886 |
| Balanced Acc | 0.8947 |
| Macro-Sensitivity | 0.8947 |
| Macro-Specificity | 0.9449 |
| Macro-Precision | 0.8889 |
| AUC-PR (Macro) | 0.9520 |
| Cohen's Kappa | 0.8343 |
| MCC | 0.8355 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.78G |
| FLOPs | 7.57G |
| Params | 21.37M |
| Pure-Inference FPS | 1003.4 (1.00 ms/张) |
| End-to-End FPS | 202.5 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8320 | 0.9470 | 0.9064 | 0.8676 | 0.9014 | 0.9665 | 0.9473 | 256 |
| Malignant_lesions | 0.8667 | 0.9371 | 0.8626 | 0.8646 | 0.9391 | 0.9669 | 0.9252 | 210 |
| body_of_stomach_normal | 0.9854 | 0.9506 | 0.8978 | 0.9395 | 0.9933 | 0.9938 | 0.9834 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=213 FP=22 TN=393 FN=43
- **Malignant_lesions**: TP=182 FP=29 TN=432 FN=28
- **body_of_stomach_normal**: TP=202 FP=23 TN=443 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 213 | 28 | 15 |
| Malignant_lesions | 20 | 182 | 8 |
| body_of_stomach_normal | 2 | 1 | 202 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9064    0.8320    0.8676       256
     Malignant_lesions     0.8626    0.8667    0.8646       210
body_of_stomach_normal     0.8978    0.9854    0.9395       205

              accuracy                         0.8897       671
             macro avg     0.8889    0.8947    0.8906       671
          weighted avg     0.8900    0.8897    0.8886       671

```