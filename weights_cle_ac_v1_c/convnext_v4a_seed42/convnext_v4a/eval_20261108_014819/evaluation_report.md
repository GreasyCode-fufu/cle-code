# convnext_v4a 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4a_seed42/convnext_v4a/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9274 |
| Accuracy | 0.9271 |
| Macro-AUROC (OvR) | 0.9908 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9276 |
| Balanced Acc | 0.9264 |
| Macro-Sensitivity | 0.9264 |
| Macro-Specificity | 0.9632 |
| Macro-Precision | 0.9291 |
| AUC-PR (Macro) | 0.9837 |
| Cohen's Kappa | 0.8903 |
| MCC | 0.8906 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 2.55G |
| FLOPs | 5.09G |
| Params | 17.62M |
| Pure-Inference FPS | 414.5 (2.41 ms/张) |
| End-to-End FPS | 176.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9348 | 0.8894 | 0.9082 | 0.9582 | 0.9879 | 0.9794 | 208 |
| Malignant_lesions | 0.9034 | 0.9575 | 0.9034 | 0.9034 | 0.9575 | 0.9859 | 0.9739 | 176 |
| body_of_stomach_normal | 0.9479 | 0.9974 | 0.9945 | 0.9707 | 0.9746 | 0.9988 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=24 TN=344 FN=15
- **Malignant_lesions**: TP=159 FP=17 TN=383 FN=17
- **body_of_stomach_normal**: TP=182 FP=1 TN=383 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 14 | 1 |
| Malignant_lesions | 17 | 159 | 0 |
| body_of_stomach_normal | 7 | 3 | 182 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8894    0.9279    0.9082       208
     Malignant_lesions     0.9034    0.9034    0.9034       176
body_of_stomach_normal     0.9945    0.9479    0.9707       192

              accuracy                         0.9271       576
             macro avg     0.9291    0.9264    0.9274       576
          weighted avg     0.9287    0.9271    0.9276       576

```