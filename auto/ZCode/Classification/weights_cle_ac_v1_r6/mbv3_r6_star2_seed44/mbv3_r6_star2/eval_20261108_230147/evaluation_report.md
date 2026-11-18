# mbv3_r6_star2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star2_seed44/mbv3_r6_star2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9682 |
| Accuracy | 0.9742 |
| Macro-AUROC (OvR) | 0.9987 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9745 |
| Balanced Acc | 0.9762 |
| Macro-Sensitivity | 0.9762 |
| Macro-Specificity | 0.9885 |
| Macro-Precision | 0.9616 |
| AUC-PR (Macro) | 0.9965 |
| Cohen's Kappa | 0.9593 |
| MCC | 0.9597 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 3141.7 (0.32 ms/张) |
| End-to-End FPS | 220.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9538 | 0.9914 | 0.9894 | 0.9713 | 0.9623 | 0.9987 | 0.9983 | 195 |
| Malignant_lesions | 0.9747 | 0.9741 | 0.8953 | 0.9333 | 0.9941 | 0.9975 | 0.9912 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=186 FP=2 TN=230 FN=9
- **Malignant_lesions**: TP=77 FP=9 TN=339 FN=2
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 186 | 9 | 0 |
| Malignant_lesions | 2 | 77 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9894    0.9538    0.9713       195
     Malignant_lesions     0.8953    0.9747    0.9333        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9742       427
             macro avg     0.9616    0.9762    0.9682       427
          weighted avg     0.9758    0.9742    0.9745       427

```