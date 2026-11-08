# convnext_v5xrcw 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5xrcw_seed43/convnext_v5xrcw/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9546 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9966 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9547 |
| Balanced Acc | 0.9559 |
| Macro-Sensitivity | 0.9559 |
| Macro-Specificity | 0.9777 |
| Macro-Precision | 0.9541 |
| AUC-PR (Macro) | 0.9931 |
| Cohen's Kappa | 0.9322 |
| MCC | 0.9326 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.69M |
| Pure-Inference FPS | 432.6 (2.31 ms/张) |
| End-to-End FPS | 179.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9183 | 0.9837 | 0.9695 | 0.9432 | 0.9551 | 0.9956 | 0.9925 | 208 |
| Malignant_lesions | 0.9545 | 0.9650 | 0.9231 | 0.9385 | 0.9797 | 0.9947 | 0.9879 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9844 | 0.9695 | 0.9820 | 0.9974 | 0.9994 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=6 TN=362 FN=17
- **Malignant_lesions**: TP=168 FP=14 TN=386 FN=8
- **body_of_stomach_normal**: TP=191 FP=6 TN=378 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 14 | 3 |
| Malignant_lesions | 5 | 168 | 3 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9695    0.9183    0.9432       208
     Malignant_lesions     0.9231    0.9545    0.9385       176
body_of_stomach_normal     0.9695    0.9948    0.9820       192

              accuracy                         0.9549       576
             macro avg     0.9541    0.9559    0.9546       576
          weighted avg     0.9553    0.9549    0.9547       576

```