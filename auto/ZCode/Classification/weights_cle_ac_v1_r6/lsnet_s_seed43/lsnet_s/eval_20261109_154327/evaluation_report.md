# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/lsnet_s_seed43/lsnet_s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9620 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9694 |
| Balanced Acc | 0.9597 |
| Macro-Sensitivity | 0.9597 |
| Macro-Specificity | 0.9839 |
| Macro-Precision | 0.9643 |
| AUC-PR (Macro) | 0.9888 |
| Cohen's Kappa | 0.9515 |
| MCC | 0.9515 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 4018.9 (0.25 ms/张) |
| End-to-End FPS | 277.3 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9698 | 0.9645 | 0.9694 | 0.9783 | 0.9958 | 0.9952 | 195 |
| Malignant_lesions | 0.9114 | 0.9856 | 0.9351 | 0.9231 | 0.9800 | 0.9933 | 0.9715 | 79 |
| body_of_stomach_normal | 0.9935 | 0.9964 | 0.9935 | 0.9935 | 0.9964 | 0.9998 | 0.9996 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=7 TN=225 FN=5
- **Malignant_lesions**: TP=72 FP=5 TN=343 FN=7
- **body_of_stomach_normal**: TP=152 FP=1 TN=273 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 5 | 0 |
| Malignant_lesions | 6 | 72 | 1 |
| body_of_stomach_normal | 1 | 0 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9645    0.9744    0.9694       195
     Malignant_lesions     0.9351    0.9114    0.9231        79
body_of_stomach_normal     0.9935    0.9935    0.9935       153

              accuracy                         0.9696       427
             macro avg     0.9643    0.9597    0.9620       427
          weighted avg     0.9694    0.9696    0.9694       427

```