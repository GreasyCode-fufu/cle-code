# thinkingvit_deit 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/thinkingvit_deit/seed42/thinkingvit_deit/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9173 |
| Accuracy | 0.9167 |
| Macro-AUROC (OvR) | 0.9900 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9175 |
| Balanced Acc | 0.9183 |
| Macro-Sensitivity | 0.9183 |
| Macro-Specificity | 0.9587 |
| Macro-Precision | 0.9193 |
| AUC-PR (Macro) | 0.9813 |
| Cohen's Kappa | 0.8750 |
| MCC | 0.8763 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 130.13M |
| FLOPs | 260.26M |
| Params | 369.41K |
| Pure-Inference FPS | 1788.9 (0.56 ms/张) |
| End-to-End FPS | 217.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8894 | 0.9538 | 0.9158 | 0.9024 | 0.9385 | 0.9843 | 0.9761 | 208 |
| Malignant_lesions | 0.9489 | 0.9250 | 0.8477 | 0.8954 | 0.9763 | 0.9865 | 0.9695 | 176 |
| body_of_stomach_normal | 0.9167 | 0.9974 | 0.9944 | 0.9539 | 0.9599 | 0.9992 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=185 FP=17 TN=351 FN=23
- **Malignant_lesions**: TP=167 FP=30 TN=370 FN=9
- **body_of_stomach_normal**: TP=176 FP=1 TN=383 FN=16

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 185 | 22 | 1 |
| Malignant_lesions | 9 | 167 | 0 |
| body_of_stomach_normal | 8 | 8 | 176 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9158    0.8894    0.9024       208
     Malignant_lesions     0.8477    0.9489    0.8954       176
body_of_stomach_normal     0.9944    0.9167    0.9539       192

              accuracy                         0.9167       576
             macro avg     0.9193    0.9183    0.9173       576
          weighted avg     0.9212    0.9167    0.9175       576

```