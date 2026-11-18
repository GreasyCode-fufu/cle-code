# thinkingvit_deit 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/thinkingvit_deit/seed44/thinkingvit_deit/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9484 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9944 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9491 |
| Balanced Acc | 0.9477 |
| Macro-Sensitivity | 0.9477 |
| Macro-Specificity | 0.9746 |
| Macro-Precision | 0.9510 |
| AUC-PR (Macro) | 0.9898 |
| Cohen's Kappa | 0.9242 |
| MCC | 0.9250 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 130.13M |
| FLOPs | 260.26M |
| Params | 369.41K |
| Pure-Inference FPS | 1789.1 (0.56 ms/张) |
| End-to-End FPS | 235.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9701 | 0.9476 | 0.9522 | 0.9754 | 0.9915 | 0.9877 | 208 |
| Malignant_lesions | 0.8864 | 0.9875 | 0.9689 | 0.9258 | 0.9518 | 0.9928 | 0.9845 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9661 | 0.9366 | 0.9673 | 1.0000 | 0.9987 | 0.9973 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=11 TN=357 FN=9
- **Malignant_lesions**: TP=156 FP=5 TN=395 FN=20
- **body_of_stomach_normal**: TP=192 FP=13 TN=371 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 5 | 4 |
| Malignant_lesions | 11 | 156 | 9 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9476    0.9567    0.9522       208
     Malignant_lesions     0.9689    0.8864    0.9258       176
body_of_stomach_normal     0.9366    1.0000    0.9673       192

              accuracy                         0.9497       576
             macro avg     0.9510    0.9477    0.9484       576
          weighted avg     0.9505    0.9497    0.9491       576

```