# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/jumbo_small_seed43/jumbo_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9546 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9946 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9550 |
| Balanced Acc | 0.9537 |
| Macro-Sensitivity | 0.9537 |
| Macro-Specificity | 0.9772 |
| Macro-Precision | 0.9559 |
| AUC-PR (Macro) | 0.9893 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9323 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 711.5 (1.41 ms/张) |
| End-to-End FPS | 203.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9565 | 0.9259 | 0.9434 | 0.9778 | 0.9923 | 0.9874 | 208 |
| Malignant_lesions | 0.9205 | 0.9750 | 0.9419 | 0.9310 | 0.9653 | 0.9917 | 0.9813 | 176 |
| body_of_stomach_normal | 0.9792 | 1.0000 | 1.0000 | 0.9895 | 0.9897 | 0.9997 | 0.9994 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=16 TN=352 FN=8
- **Malignant_lesions**: TP=162 FP=10 TN=390 FN=14
- **body_of_stomach_normal**: TP=188 FP=0 TN=384 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 8 | 0 |
| Malignant_lesions | 14 | 162 | 0 |
| body_of_stomach_normal | 2 | 2 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9259    0.9615    0.9434       208
     Malignant_lesions     0.9419    0.9205    0.9310       176
body_of_stomach_normal     1.0000    0.9792    0.9895       192

              accuracy                         0.9549       576
             macro avg     0.9559    0.9537    0.9546       576
          weighted avg     0.9555    0.9549    0.9550       576

```