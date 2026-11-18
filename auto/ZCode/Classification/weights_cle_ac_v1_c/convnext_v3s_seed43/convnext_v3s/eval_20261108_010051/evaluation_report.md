# convnext_v3s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v3s_seed43/convnext_v3s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9508 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9958 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9511 |
| Balanced Acc | 0.9500 |
| Macro-Sensitivity | 0.9500 |
| Macro-Specificity | 0.9754 |
| Macro-Precision | 0.9523 |
| AUC-PR (Macro) | 0.9918 |
| Cohen's Kappa | 0.9268 |
| MCC | 0.9272 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.65G |
| FLOPs | 9.31G |
| Params | 28.81M |
| Pure-Inference FPS | 375.8 (2.66 ms/张) |
| End-to-End FPS | 168.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9592 | 0.9296 | 0.9406 | 0.9725 | 0.9936 | 0.9897 | 208 |
| Malignant_lesions | 0.9034 | 0.9825 | 0.9578 | 0.9298 | 0.9585 | 0.9945 | 0.9869 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9844 | 0.9695 | 0.9820 | 0.9974 | 0.9993 | 0.9989 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=15 TN=353 FN=10
- **Malignant_lesions**: TP=159 FP=7 TN=393 FN=17
- **body_of_stomach_normal**: TP=191 FP=6 TN=378 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 14 | 159 | 3 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9296    0.9519    0.9406       208
     Malignant_lesions     0.9578    0.9034    0.9298       176
body_of_stomach_normal     0.9695    0.9948    0.9820       192

              accuracy                         0.9514       576
             macro avg     0.9523    0.9500    0.9508       576
          weighted avg     0.9515    0.9514    0.9511       576

```