# mbv3_lpft 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_lpft_seed45/mbv3_lpft/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9403 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9910 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9406 |
| Balanced Acc | 0.9391 |
| Macro-Sensitivity | 0.9391 |
| Macro-Specificity | 0.9700 |
| Macro-Precision | 0.9427 |
| AUC-PR (Macro) | 0.9824 |
| Cohen's Kappa | 0.9111 |
| MCC | 0.9117 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 3399.9 (0.29 ms/张) |
| End-to-End FPS | 132.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9457 | 0.9078 | 0.9271 | 0.9694 | 0.9874 | 0.9804 | 208 |
| Malignant_lesions | 0.8807 | 0.9800 | 0.9509 | 0.9145 | 0.9492 | 0.9862 | 0.9680 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9844 | 0.9694 | 0.9794 | 0.9947 | 0.9995 | 0.9989 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=20 TN=348 FN=11
- **Malignant_lesions**: TP=155 FP=8 TN=392 FN=21
- **body_of_stomach_normal**: TP=190 FP=6 TN=378 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 8 | 3 |
| Malignant_lesions | 18 | 155 | 3 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9078    0.9471    0.9271       208
     Malignant_lesions     0.9509    0.8807    0.9145       176
body_of_stomach_normal     0.9694    0.9896    0.9794       192

              accuracy                         0.9410       576
             macro avg     0.9427    0.9391    0.9403       576
          weighted avg     0.9415    0.9410    0.9406       576

```