# mbv3_r6_mlp256 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_mlp256_seed44/mbv3_r6_mlp256/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9642 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9971 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9718 |
| Balanced Acc | 0.9619 |
| Macro-Sensitivity | 0.9619 |
| Macro-Specificity | 0.9852 |
| Macro-Precision | 0.9665 |
| AUC-PR (Macro) | 0.9909 |
| Cohen's Kappa | 0.9552 |
| MCC | 0.9553 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.93M |
| FLOPs | 463.85M |
| Params | 3.22M |
| Pure-Inference FPS | 2844.3 (0.35 ms/张) |
| End-to-End FPS | 223.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9698 | 0.9645 | 0.9694 | 0.9783 | 0.9968 | 0.9963 | 195 |
| Malignant_lesions | 0.9114 | 0.9856 | 0.9351 | 0.9231 | 0.9800 | 0.9945 | 0.9764 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=7 TN=225 FN=5
- **Malignant_lesions**: TP=72 FP=5 TN=343 FN=7
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 5 | 0 |
| Malignant_lesions | 7 | 72 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9645    0.9744    0.9694       195
     Malignant_lesions     0.9351    0.9114    0.9231        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9719       427
             macro avg     0.9665    0.9619    0.9642       427
          weighted avg     0.9718    0.9719    0.9718       427

```