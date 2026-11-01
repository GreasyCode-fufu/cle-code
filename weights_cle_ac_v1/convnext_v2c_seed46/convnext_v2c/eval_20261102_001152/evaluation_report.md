# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2c_seed46/convnext_v2c/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9595 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9953 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9599 |
| Balanced Acc | 0.9595 |
| Macro-Sensitivity | 0.9595 |
| Macro-Specificity | 0.9800 |
| Macro-Precision | 0.9597 |
| AUC-PR (Macro) | 0.9905 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9400 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 659.5 (1.52 ms/张) |
| End-to-End FPS | 203.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9755 | 0.9565 | 0.9542 | 0.9729 | 0.9938 | 0.9901 | 208 |
| Malignant_lesions | 0.9318 | 0.9775 | 0.9480 | 0.9398 | 0.9702 | 0.9926 | 0.9821 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9870 | 0.9745 | 0.9845 | 0.9974 | 0.9997 | 0.9994 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=9 TN=359 FN=10
- **Malignant_lesions**: TP=164 FP=9 TN=391 FN=12
- **body_of_stomach_normal**: TP=191 FP=5 TN=379 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 9 | 1 |
| Malignant_lesions | 8 | 164 | 4 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9565    0.9519    0.9542       208
     Malignant_lesions     0.9480    0.9318    0.9398       176
body_of_stomach_normal     0.9745    0.9948    0.9845       192

              accuracy                         0.9601       576
             macro avg     0.9597    0.9595    0.9595       576
          weighted avg     0.9599    0.9601    0.9599       576

```