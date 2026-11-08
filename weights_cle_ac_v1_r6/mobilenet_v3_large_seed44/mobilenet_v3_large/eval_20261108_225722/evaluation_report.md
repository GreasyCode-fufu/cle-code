# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mobilenet_v3_large_seed44/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9342 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9909 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9343 |
| Balanced Acc | 0.9310 |
| Macro-Sensitivity | 0.9310 |
| Macro-Specificity | 0.9660 |
| Macro-Precision | 0.9415 |
| AUC-PR (Macro) | 0.9829 |
| Cohen's Kappa | 0.9005 |
| MCC | 0.9026 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 4059.8 (0.25 ms/张) |
| End-to-End FPS | 228.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9158 | 0.8670 | 0.9161 | 0.9825 | 0.9874 | 0.9774 | 208 |
| Malignant_lesions | 0.8636 | 0.9875 | 0.9682 | 0.9129 | 0.9427 | 0.9908 | 0.9798 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9948 | 0.9892 | 0.9735 | 0.9795 | 0.9944 | 0.9917 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=31 TN=337 FN=6
- **Malignant_lesions**: TP=152 FP=5 TN=395 FN=24
- **body_of_stomach_normal**: TP=184 FP=2 TN=382 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 23 | 152 | 1 |
| body_of_stomach_normal | 8 | 0 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8670    0.9712    0.9161       208
     Malignant_lesions     0.9682    0.8636    0.9129       176
body_of_stomach_normal     0.9892    0.9583    0.9735       192

              accuracy                         0.9340       576
             macro avg     0.9415    0.9310    0.9342       576
          weighted avg     0.9386    0.9340    0.9343       576

```