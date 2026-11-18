# convnext_v4b 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4b_seed43/convnext_v4b/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9392 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9932 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9394 |
| Balanced Acc | 0.9392 |
| Macro-Sensitivity | 0.9392 |
| Macro-Specificity | 0.9695 |
| Macro-Precision | 0.9410 |
| AUC-PR (Macro) | 0.9880 |
| Cohen's Kappa | 0.9086 |
| MCC | 0.9094 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.93G |
| FLOPs | 7.87G |
| Params | 14.08M |
| Pure-Inference FPS | 380.5 (2.63 ms/张) |
| End-to-End FPS | 170.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9538 | 0.9213 | 0.9387 | 0.9750 | 0.9906 | 0.9855 | 208 |
| Malignant_lesions | 0.9545 | 0.9600 | 0.9130 | 0.9333 | 0.9796 | 0.9909 | 0.9816 | 176 |
| body_of_stomach_normal | 0.9062 | 0.9948 | 0.9886 | 0.9457 | 0.9550 | 0.9983 | 0.9968 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=17 TN=351 FN=9
- **Malignant_lesions**: TP=168 FP=16 TN=384 FN=8
- **body_of_stomach_normal**: TP=174 FP=2 TN=382 FN=18

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 7 | 168 | 1 |
| body_of_stomach_normal | 10 | 8 | 174 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9213    0.9567    0.9387       208
     Malignant_lesions     0.9130    0.9545    0.9333       176
body_of_stomach_normal     0.9886    0.9062    0.9457       192

              accuracy                         0.9392       576
             macro avg     0.9410    0.9392    0.9392       576
          weighted avg     0.9412    0.9392    0.9394       576

```