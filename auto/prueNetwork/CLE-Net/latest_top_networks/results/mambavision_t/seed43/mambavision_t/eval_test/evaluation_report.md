# mambavision_t 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/mambavision_t/seed43/mambavision_t/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9629 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9954 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9633 |
| Balanced Acc | 0.9626 |
| Macro-Sensitivity | 0.9626 |
| Macro-Specificity | 0.9816 |
| Macro-Precision | 0.9639 |
| AUC-PR (Macro) | 0.9913 |
| Cohen's Kappa | 0.9452 |
| MCC | 0.9454 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.47G |
| FLOPs | 8.94G |
| Params | 31.14M |
| Pure-Inference FPS | 2572.7 (0.39 ms/张) |
| End-to-End FPS | 243.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9783 | 0.9615 | 0.9615 | 0.9783 | 0.9930 | 0.9892 | 208 |
| Malignant_lesions | 0.9261 | 0.9875 | 0.9702 | 0.9477 | 0.9681 | 0.9942 | 0.9870 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9792 | 0.9600 | 0.9796 | 1.0000 | 0.9989 | 0.9976 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=8 TN=360 FN=8
- **Malignant_lesions**: TP=163 FP=5 TN=395 FN=13
- **body_of_stomach_normal**: TP=192 FP=8 TN=376 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 5 | 3 |
| Malignant_lesions | 8 | 163 | 5 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9615    0.9615    0.9615       208
     Malignant_lesions     0.9702    0.9261    0.9477       176
body_of_stomach_normal     0.9600    1.0000    0.9796       192

              accuracy                         0.9635       576
             macro avg     0.9639    0.9626    0.9629       576
          weighted avg     0.9637    0.9635    0.9633       576

```