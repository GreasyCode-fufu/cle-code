# mambaout_tiny 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/mambaout_tiny/seed44/mambaout_tiny/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9512 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9937 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9515 |
| Balanced Acc | 0.9506 |
| Macro-Sensitivity | 0.9506 |
| Macro-Specificity | 0.9755 |
| Macro-Precision | 0.9522 |
| AUC-PR (Macro) | 0.9878 |
| Cohen's Kappa | 0.9269 |
| MCC | 0.9271 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.48G |
| FLOPs | 8.96G |
| Params | 24.25M |
| Pure-Inference FPS | 357.6 (2.80 ms/张) |
| End-to-End FPS | 15.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9592 | 0.9302 | 0.9456 | 0.9778 | 0.9926 | 0.9885 | 208 |
| Malignant_lesions | 0.9318 | 0.9725 | 0.9371 | 0.9345 | 0.9701 | 0.9892 | 0.9763 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9948 | 0.9892 | 0.9735 | 0.9795 | 0.9993 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=15 TN=353 FN=8
- **Malignant_lesions**: TP=164 FP=11 TN=389 FN=12
- **body_of_stomach_normal**: TP=184 FP=2 TN=382 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 7 | 1 |
| Malignant_lesions | 11 | 164 | 1 |
| body_of_stomach_normal | 4 | 4 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9302    0.9615    0.9456       208
     Malignant_lesions     0.9371    0.9318    0.9345       176
body_of_stomach_normal     0.9892    0.9583    0.9735       192

              accuracy                         0.9514       576
             macro avg     0.9522    0.9506    0.9512       576
          weighted avg     0.9520    0.9514    0.9515       576

```