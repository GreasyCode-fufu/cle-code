# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/pvt_tiny_seed42/pvt_tiny/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8846 |
| Accuracy | 0.8906 |
| Macro-AUROC (OvR) | 0.9887 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8866 |
| Balanced Acc | 0.8842 |
| Macro-Sensitivity | 0.8842 |
| Macro-Specificity | 0.9441 |
| Macro-Precision | 0.9012 |
| AUC-PR (Macro) | 0.9787 |
| Cohen's Kappa | 0.8347 |
| MCC | 0.8409 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 730.4 (1.37 ms/张) |
| End-to-End FPS | 207.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9022 | 0.8448 | 0.8909 | 0.9651 | 0.9825 | 0.9754 | 208 |
| Malignant_lesions | 0.7102 | 0.9875 | 0.9615 | 0.8170 | 0.8857 | 0.9846 | 0.9630 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9427 | 0.8972 | 0.9458 | 1.0000 | 0.9989 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=36 TN=332 FN=12
- **Malignant_lesions**: TP=125 FP=5 TN=395 FN=51
- **body_of_stomach_normal**: TP=192 FP=22 TN=362 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 5 | 7 |
| Malignant_lesions | 36 | 125 | 15 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8448    0.9423    0.8909       208
     Malignant_lesions     0.9615    0.7102    0.8170       176
body_of_stomach_normal     0.8972    1.0000    0.9458       192

              accuracy                         0.8906       576
             macro avg     0.9012    0.8842    0.8846       576
          weighted avg     0.8979    0.8906    0.8866       576

```