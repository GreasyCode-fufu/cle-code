# mambaout_tiny 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/mambaout_tiny/seed42/mambaout_tiny/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9497 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9499 |
| Balanced Acc | 0.9477 |
| Macro-Sensitivity | 0.9477 |
| Macro-Specificity | 0.9742 |
| Macro-Precision | 0.9536 |
| AUC-PR (Macro) | 0.9903 |
| Cohen's Kappa | 0.9242 |
| MCC | 0.9252 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.48G |
| FLOPs | 8.96G |
| Params | 24.25M |
| Pure-Inference FPS | 341.1 (2.93 ms/张) |
| End-to-End FPS | 16.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9402 | 0.9027 | 0.9401 | 0.9886 | 0.9941 | 0.9912 | 208 |
| Malignant_lesions | 0.9091 | 0.9825 | 0.9581 | 0.9329 | 0.9609 | 0.9911 | 0.9807 | 176 |
| body_of_stomach_normal | 0.9531 | 1.0000 | 1.0000 | 0.9760 | 0.9771 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=22 TN=346 FN=4
- **Malignant_lesions**: TP=160 FP=7 TN=393 FN=16
- **body_of_stomach_normal**: TP=183 FP=0 TN=384 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 4 | 0 |
| Malignant_lesions | 16 | 160 | 0 |
| body_of_stomach_normal | 6 | 3 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9027    0.9808    0.9401       208
     Malignant_lesions     0.9581    0.9091    0.9329       176
body_of_stomach_normal     1.0000    0.9531    0.9760       192

              accuracy                         0.9497       576
             macro avg     0.9536    0.9477    0.9497       576
          weighted avg     0.9520    0.9497    0.9499       576

```