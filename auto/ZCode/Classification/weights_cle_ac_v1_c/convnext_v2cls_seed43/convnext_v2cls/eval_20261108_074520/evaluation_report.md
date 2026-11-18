# convnext_v2cls 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2cls_seed43/convnext_v2cls/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9499 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9944 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9508 |
| Balanced Acc | 0.9482 |
| Macro-Sensitivity | 0.9482 |
| Macro-Specificity | 0.9751 |
| Macro-Precision | 0.9548 |
| AUC-PR (Macro) | 0.9878 |
| Cohen's Kappa | 0.9267 |
| MCC | 0.9281 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 465.2 (2.15 ms/张) |
| End-to-End FPS | 186.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9856 | 0.9511 | 0.9193 | 0.9513 | 0.9915 | 0.9911 | 0.9841 | 208 |
| Malignant_lesions | 0.8693 | 0.9925 | 0.9808 | 0.9217 | 0.9452 | 0.9931 | 0.9817 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9818 | 0.9645 | 0.9769 | 0.9947 | 0.9989 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=205 FP=18 TN=350 FN=3
- **Malignant_lesions**: TP=153 FP=3 TN=397 FN=23
- **body_of_stomach_normal**: TP=190 FP=7 TN=377 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 205 | 3 | 0 |
| Malignant_lesions | 16 | 153 | 7 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9193    0.9856    0.9513       208
     Malignant_lesions     0.9808    0.8693    0.9217       176
body_of_stomach_normal     0.9645    0.9896    0.9769       192

              accuracy                         0.9514       576
             macro avg     0.9548    0.9482    0.9499       576
          weighted avg     0.9531    0.9514    0.9508       576

```