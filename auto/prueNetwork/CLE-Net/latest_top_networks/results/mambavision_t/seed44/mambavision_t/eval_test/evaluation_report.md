# mambavision_t 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/mambavision_t/seed44/mambavision_t/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9359 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9934 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9366 |
| Balanced Acc | 0.9336 |
| Macro-Sensitivity | 0.9336 |
| Macro-Specificity | 0.9678 |
| Macro-Precision | 0.9439 |
| AUC-PR (Macro) | 0.9882 |
| Cohen's Kappa | 0.9057 |
| MCC | 0.9082 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.47G |
| FLOPs | 8.94G |
| Params | 31.14M |
| Pure-Inference FPS | 2568.7 (0.39 ms/张) |
| End-to-End FPS | 205.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9266 | 0.8821 | 0.9245 | 0.9827 | 0.9898 | 0.9850 | 208 |
| Malignant_lesions | 0.8295 | 0.9925 | 0.9799 | 0.8985 | 0.9297 | 0.9912 | 0.9812 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9844 | 0.9697 | 0.9846 | 1.0000 | 0.9993 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=27 TN=341 FN=6
- **Malignant_lesions**: TP=146 FP=3 TN=397 FN=30
- **body_of_stomach_normal**: TP=192 FP=6 TN=378 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 3 | 3 |
| Malignant_lesions | 27 | 146 | 3 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8821    0.9712    0.9245       208
     Malignant_lesions     0.9799    0.8295    0.8985       176
body_of_stomach_normal     0.9697    1.0000    0.9846       192

              accuracy                         0.9375       576
             macro avg     0.9439    0.9336    0.9359       576
          weighted avg     0.9412    0.9375    0.9366       576

```