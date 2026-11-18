# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/ghostnet_seed42/ghostnet/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9321 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9912 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9329 |
| Balanced Acc | 0.9304 |
| Macro-Sensitivity | 0.9304 |
| Macro-Specificity | 0.9662 |
| Macro-Precision | 0.9389 |
| AUC-PR (Macro) | 0.9827 |
| Cohen's Kappa | 0.9005 |
| MCC | 0.9026 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 5059.3 (0.20 ms/张) |
| End-to-End FPS | 164.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9348 | 0.8929 | 0.9259 | 0.9773 | 0.9867 | 0.9769 | 208 |
| Malignant_lesions | 0.8295 | 0.9900 | 0.9733 | 0.8957 | 0.9296 | 0.9878 | 0.9729 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9740 | 0.9505 | 0.9746 | 1.0000 | 0.9992 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=24 TN=344 FN=8
- **Malignant_lesions**: TP=146 FP=4 TN=396 FN=30
- **body_of_stomach_normal**: TP=192 FP=10 TN=374 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 4 | 4 |
| Malignant_lesions | 24 | 146 | 6 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8929    0.9615    0.9259       208
     Malignant_lesions     0.9733    0.8295    0.8957       176
body_of_stomach_normal     0.9505    1.0000    0.9746       192

              accuracy                         0.9340       576
             macro avg     0.9389    0.9304    0.9321       576
          weighted avg     0.9367    0.9340    0.9329       576

```