# mambavision_t 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/mambavision_t/seed42/mambavision_t/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9359 |
| Accuracy | 0.9358 |
| Macro-AUROC (OvR) | 0.9920 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9361 |
| Balanced Acc | 0.9331 |
| Macro-Sensitivity | 0.9331 |
| Macro-Specificity | 0.9670 |
| Macro-Precision | 0.9418 |
| AUC-PR (Macro) | 0.9857 |
| Cohen's Kappa | 0.9032 |
| MCC | 0.9048 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.47G |
| FLOPs | 8.94G |
| Params | 31.14M |
| Pure-Inference FPS | 2574.7 (0.39 ms/张) |
| End-to-End FPS | 226.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9212 | 0.8745 | 0.9203 | 0.9826 | 0.9906 | 0.9838 | 208 |
| Malignant_lesions | 0.8750 | 0.9825 | 0.9565 | 0.9139 | 0.9470 | 0.9869 | 0.9759 | 176 |
| body_of_stomach_normal | 0.9531 | 0.9974 | 0.9946 | 0.9734 | 0.9770 | 0.9985 | 0.9974 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=29 TN=339 FN=6
- **Malignant_lesions**: TP=154 FP=7 TN=393 FN=22
- **body_of_stomach_normal**: TP=183 FP=1 TN=383 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 22 | 154 | 0 |
| body_of_stomach_normal | 7 | 2 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8745    0.9712    0.9203       208
     Malignant_lesions     0.9565    0.8750    0.9139       176
body_of_stomach_normal     0.9946    0.9531    0.9734       192

              accuracy                         0.9358       576
             macro avg     0.9418    0.9331    0.9359       576
          weighted avg     0.9396    0.9358    0.9361       576

```