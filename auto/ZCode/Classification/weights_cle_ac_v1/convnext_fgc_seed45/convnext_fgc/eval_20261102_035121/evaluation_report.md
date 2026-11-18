# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_seed45/convnext_fgc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9614 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9969 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9618 |
| Balanced Acc | 0.9610 |
| Macro-Sensitivity | 0.9610 |
| Macro-Specificity | 0.9808 |
| Macro-Precision | 0.9618 |
| AUC-PR (Macro) | 0.9939 |
| Cohen's Kappa | 0.9426 |
| MCC | 0.9426 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 730.8 (1.37 ms/张) |
| End-to-End FPS | 214.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9701 | 0.9479 | 0.9547 | 0.9781 | 0.9959 | 0.9933 | 208 |
| Malignant_lesions | 0.9318 | 0.9775 | 0.9480 | 0.9398 | 0.9702 | 0.9950 | 0.9891 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9948 | 0.9896 | 0.9896 | 0.9948 | 0.9996 | 0.9993 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=11 TN=357 FN=8
- **Malignant_lesions**: TP=164 FP=9 TN=391 FN=12
- **body_of_stomach_normal**: TP=190 FP=2 TN=382 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 8 | 0 |
| Malignant_lesions | 10 | 164 | 2 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9479    0.9615    0.9547       208
     Malignant_lesions     0.9480    0.9318    0.9398       176
body_of_stomach_normal     0.9896    0.9896    0.9896       192

              accuracy                         0.9618       576
             macro avg     0.9618    0.9610    0.9614       576
          weighted avg     0.9618    0.9618    0.9618       576

```