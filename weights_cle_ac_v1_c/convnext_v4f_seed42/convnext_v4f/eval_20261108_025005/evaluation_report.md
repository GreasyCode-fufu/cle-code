# convnext_v4f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4f_seed42/convnext_v4f/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9792 |
| Accuracy | 0.9836 |
| Macro-AUROC (OvR) | 0.9987 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9836 |
| Balanced Acc | 0.9780 |
| Macro-Sensitivity | 0.9780 |
| Macro-Specificity | 0.9914 |
| Macro-Precision | 0.9804 |
| AUC-PR (Macro) | 0.9955 |
| Cohen's Kappa | 0.9739 |
| MCC | 0.9739 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 1402.9 (0.71 ms/张) |
| End-to-End FPS | 256.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9828 | 0.9796 | 0.9821 | 0.9870 | 0.9986 | 0.9984 | 195 |
| Malignant_lesions | 0.9494 | 0.9914 | 0.9615 | 0.9554 | 0.9885 | 0.9975 | 0.9880 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=4 TN=228 FN=3
- **Malignant_lesions**: TP=75 FP=3 TN=345 FN=4
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 3 | 0 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9796    0.9846    0.9821       195
     Malignant_lesions     0.9615    0.9494    0.9554        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9836       427
             macro avg     0.9804    0.9780    0.9792       427
          weighted avg     0.9836    0.9836    0.9836       427

```