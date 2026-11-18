# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-25/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9033 |
| Accuracy | 0.9016 |
| Macro-AUROC (OvR) | 0.9823 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9008 |
| Balanced Acc | 0.9094 |
| Macro-Sensitivity | 0.9094 |
| Macro-Specificity | 0.9516 |
| Macro-Precision | 0.9042 |
| AUC-PR (Macro) | 0.9682 |
| Cohen's Kappa | 0.8526 |
| MCC | 0.8562 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.78G |
| FLOPs | 7.57G |
| Params | 21.37M |
| Pure-Inference FPS | 1003.0 (1.00 ms/张) |
| End-to-End FPS | 213.2 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8047 | 0.9759 | 0.9537 | 0.8729 | 0.8901 | 0.9759 | 0.9674 | 256 |
| Malignant_lesions | 0.9333 | 0.9089 | 0.8235 | 0.8750 | 0.9677 | 0.9735 | 0.9429 | 210 |
| body_of_stomach_normal | 0.9902 | 0.9700 | 0.9355 | 0.9621 | 0.9956 | 0.9975 | 0.9941 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=206 FP=10 TN=405 FN=50
- **Malignant_lesions**: TP=196 FP=42 TN=419 FN=14
- **body_of_stomach_normal**: TP=203 FP=14 TN=452 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 206 | 41 | 9 |
| Malignant_lesions | 9 | 196 | 5 |
| body_of_stomach_normal | 1 | 1 | 203 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9537    0.8047    0.8729       256
     Malignant_lesions     0.8235    0.9333    0.8750       210
body_of_stomach_normal     0.9355    0.9902    0.9621       205

              accuracy                         0.9016       671
             macro avg     0.9042    0.9094    0.9033       671
          weighted avg     0.9074    0.9016    0.9008       671

```