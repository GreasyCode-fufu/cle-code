# resnet34 评估报告

- 权重: `weights_3class_hq_seed43/resnet34/resnet34_3class_hq_seed43_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8901 |
| Accuracy | 0.8882 |
| Macro-AUROC (OvR) | 0.9786 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8882 |
| Balanced Acc | 0.8942 |
| Macro-Sensitivity | 0.8942 |
| Macro-Specificity | 0.9447 |
| Macro-Precision | 0.8913 |
| AUC-PR (Macro) | 0.9636 |
| Cohen's Kappa | 0.8323 |
| MCC | 0.8351 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3781.3 (0.26 ms/张) |
| End-to-End FPS | 255.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8125 | 0.9639 | 0.9327 | 0.8685 | 0.8929 | 0.9712 | 0.9605 | 256 |
| Malignant_lesions | 0.9238 | 0.9002 | 0.8083 | 0.8622 | 0.9629 | 0.9719 | 0.9474 | 210 |
| body_of_stomach_normal | 0.9463 | 0.9700 | 0.9327 | 0.9395 | 0.9762 | 0.9926 | 0.9829 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=208 FP=15 TN=400 FN=48
- **Malignant_lesions**: TP=194 FP=46 TN=415 FN=16
- **body_of_stomach_normal**: TP=194 FP=14 TN=452 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 208 | 37 | 11 |
| Malignant_lesions | 13 | 194 | 3 |
| body_of_stomach_normal | 2 | 9 | 194 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9327    0.8125    0.8685       256
     Malignant_lesions     0.8083    0.9238    0.8622       210
body_of_stomach_normal     0.9327    0.9463    0.9395       205

              accuracy                         0.8882       671
             macro avg     0.8913    0.8942    0.8901       671
          weighted avg     0.8938    0.8882    0.8882       671

```