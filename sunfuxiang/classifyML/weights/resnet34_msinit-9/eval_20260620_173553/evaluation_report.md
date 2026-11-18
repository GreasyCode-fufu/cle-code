# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-9/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9144 |
| Accuracy | 0.9136 |
| Macro-AUROC (OvR) | 0.9720 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9129 |
| Balanced Acc | 0.9198 |
| Macro-Sensitivity | 0.9198 |
| Macro-Specificity | 0.9576 |
| Macro-Precision | 0.9140 |
| AUC-PR (Macro) | 0.9543 |
| Cohen's Kappa | 0.8704 |
| MCC | 0.8729 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.71G |
| FLOPs | 9.42G |
| Params | 24.58M |
| Pure-Inference FPS | 3053.8 (0.33 ms/张) |
| End-to-End FPS | 228.7 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8359 | 0.9807 | 0.9640 | 0.8954 | 0.9065 | 0.9574 | 0.9521 | 256 |
| Malignant_lesions | 0.9381 | 0.9328 | 0.8640 | 0.8995 | 0.9707 | 0.9674 | 0.9324 | 210 |
| body_of_stomach_normal | 0.9854 | 0.9592 | 0.9140 | 0.9484 | 0.9933 | 0.9913 | 0.9784 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=214 FP=8 TN=407 FN=42
- **Malignant_lesions**: TP=197 FP=31 TN=430 FN=13
- **body_of_stomach_normal**: TP=202 FP=19 TN=447 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 214 | 29 | 13 |
| Malignant_lesions | 7 | 197 | 6 |
| body_of_stomach_normal | 1 | 2 | 202 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9640    0.8359    0.8954       256
     Malignant_lesions     0.8640    0.9381    0.8995       210
body_of_stomach_normal     0.9140    0.9854    0.9484       205

              accuracy                         0.9136       671
             macro avg     0.9140    0.9198    0.9144       671
          weighted avg     0.9174    0.9136    0.9129       671

```