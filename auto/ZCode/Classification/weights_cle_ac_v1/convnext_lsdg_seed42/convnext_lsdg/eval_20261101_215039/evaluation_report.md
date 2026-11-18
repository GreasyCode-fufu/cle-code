# convnext_lsdg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_lsdg_seed42/convnext_lsdg/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9492 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9947 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9497 |
| Balanced Acc | 0.9488 |
| Macro-Sensitivity | 0.9488 |
| Macro-Specificity | 0.9747 |
| Macro-Precision | 0.9498 |
| AUC-PR (Macro) | 0.9898 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9243 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.03M |
| Pure-Inference FPS | 759.5 (1.32 ms/张) |
| End-to-End FPS | 215.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9592 | 0.9292 | 0.9381 | 0.9698 | 0.9930 | 0.9884 | 208 |
| Malignant_lesions | 0.9148 | 0.9700 | 0.9306 | 0.9226 | 0.9628 | 0.9916 | 0.9822 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9948 | 0.9895 | 0.9869 | 0.9922 | 0.9995 | 0.9989 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=15 TN=353 FN=11
- **Malignant_lesions**: TP=161 FP=12 TN=388 FN=15
- **body_of_stomach_normal**: TP=189 FP=2 TN=382 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 14 | 161 | 1 |
| body_of_stomach_normal | 1 | 2 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9292    0.9471    0.9381       208
     Malignant_lesions     0.9306    0.9148    0.9226       176
body_of_stomach_normal     0.9895    0.9844    0.9869       192

              accuracy                         0.9497       576
             macro avg     0.9498    0.9488    0.9492       576
          weighted avg     0.9498    0.9497    0.9497       576

```