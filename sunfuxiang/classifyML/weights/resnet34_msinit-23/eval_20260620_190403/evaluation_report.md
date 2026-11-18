# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-23/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8944 |
| Accuracy | 0.8927 |
| Macro-AUROC (OvR) | 0.9744 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8923 |
| Balanced Acc | 0.8974 |
| Macro-Sensitivity | 0.8974 |
| Macro-Specificity | 0.9463 |
| Macro-Precision | 0.8930 |
| AUC-PR (Macro) | 0.9559 |
| Cohen's Kappa | 0.8387 |
| MCC | 0.8396 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 5.34G |
| FLOPs | 10.68G |
| Params | 24.81M |
| Pure-Inference FPS | 2212.3 (0.45 ms/张) |
| End-to-End FPS | 225.6 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8359 | 0.9470 | 0.9068 | 0.8699 | 0.9034 | 0.9610 | 0.9535 | 256 |
| Malignant_lesions | 0.8952 | 0.9241 | 0.8430 | 0.8684 | 0.9509 | 0.9699 | 0.9335 | 210 |
| body_of_stomach_normal | 0.9610 | 0.9678 | 0.9292 | 0.9448 | 0.9826 | 0.9922 | 0.9805 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=214 FP=22 TN=393 FN=42
- **Malignant_lesions**: TP=188 FP=35 TN=426 FN=22
- **body_of_stomach_normal**: TP=197 FP=15 TN=451 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 214 | 29 | 13 |
| Malignant_lesions | 20 | 188 | 2 |
| body_of_stomach_normal | 2 | 6 | 197 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9068    0.8359    0.8699       256
     Malignant_lesions     0.8430    0.8952    0.8684       210
body_of_stomach_normal     0.9292    0.9610    0.9448       205

              accuracy                         0.8927       671
             macro avg     0.8930    0.8974    0.8944       671
          weighted avg     0.8937    0.8927    0.8923       671

```