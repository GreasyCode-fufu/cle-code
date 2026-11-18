# mbv3_r6_star2x 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star2x_seed45/mbv3_r6_star2x/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9761 |
| Accuracy | 0.9813 |
| Macro-AUROC (OvR) | 0.9976 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9812 |
| Balanced Acc | 0.9738 |
| Macro-Sensitivity | 0.9738 |
| Macro-Specificity | 0.9899 |
| Macro-Precision | 0.9786 |
| AUC-PR (Macro) | 0.9933 |
| Cohen's Kappa | 0.9701 |
| MCC | 0.9702 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.10M |
| FLOPs | 464.20M |
| Params | 3.39M |
| Pure-Inference FPS | 2963.3 (0.34 ms/张) |
| End-to-End FPS | 69.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9784 | 0.9746 | 0.9796 | 0.9870 | 0.9973 | 0.9967 | 195 |
| Malignant_lesions | 0.9367 | 0.9914 | 0.9610 | 0.9487 | 0.9857 | 0.9954 | 0.9831 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=5 TN=227 FN=3
- **Malignant_lesions**: TP=74 FP=3 TN=345 FN=5
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 3 | 0 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9746    0.9846    0.9796       195
     Malignant_lesions     0.9610    0.9367    0.9487        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9813       427
             macro avg     0.9786    0.9738    0.9761       427
          weighted avg     0.9812    0.9813    0.9812       427

```