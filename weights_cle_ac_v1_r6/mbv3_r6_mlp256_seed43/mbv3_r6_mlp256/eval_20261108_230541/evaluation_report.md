# mbv3_r6_mlp256 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_mlp256_seed43/mbv3_r6_mlp256/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9502 |
| Accuracy | 0.9602 |
| Macro-AUROC (OvR) | 0.9971 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9604 |
| Balanced Acc | 0.9534 |
| Macro-Sensitivity | 0.9534 |
| Macro-Specificity | 0.9804 |
| Macro-Precision | 0.9472 |
| AUC-PR (Macro) | 0.9913 |
| Cohen's Kappa | 0.9369 |
| MCC | 0.9370 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.93M |
| FLOPs | 463.85M |
| Params | 3.22M |
| Pure-Inference FPS | 3278.7 (0.30 ms/张) |
| End-to-End FPS | 216.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9487 | 0.9698 | 0.9635 | 0.9561 | 0.9574 | 0.9968 | 0.9962 | 195 |
| Malignant_lesions | 0.9114 | 0.9713 | 0.8780 | 0.8944 | 0.9797 | 0.9945 | 0.9775 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=185 FP=7 TN=225 FN=10
- **Malignant_lesions**: TP=72 FP=10 TN=338 FN=7
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 185 | 10 | 0 |
| Malignant_lesions | 7 | 72 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9635    0.9487    0.9561       195
     Malignant_lesions     0.8780    0.9114    0.8944        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9602       427
             macro avg     0.9472    0.9534    0.9502       427
          weighted avg     0.9608    0.9602    0.9604       427

```