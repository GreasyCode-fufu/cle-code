# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-24/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9088 |
| Accuracy | 0.9076 |
| Macro-AUROC (OvR) | 0.9774 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9072 |
| Balanced Acc | 0.9125 |
| Macro-Sensitivity | 0.9125 |
| Macro-Specificity | 0.9542 |
| Macro-Precision | 0.9077 |
| AUC-PR (Macro) | 0.9558 |
| Cohen's Kappa | 0.8612 |
| MCC | 0.8626 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.78G |
| FLOPs | 7.57G |
| Params | 21.37M |
| Pure-Inference FPS | 1000.6 (1.00 ms/张) |
| End-to-End FPS | 201.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8477 | 0.9663 | 0.9394 | 0.8912 | 0.9114 | 0.9675 | 0.9604 | 256 |
| Malignant_lesions | 0.9143 | 0.9284 | 0.8533 | 0.8828 | 0.9596 | 0.9707 | 0.9161 | 210 |
| body_of_stomach_normal | 0.9756 | 0.9678 | 0.9302 | 0.9524 | 0.9890 | 0.9940 | 0.9908 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=217 FP=14 TN=401 FN=39
- **Malignant_lesions**: TP=192 FP=33 TN=428 FN=18
- **body_of_stomach_normal**: TP=200 FP=15 TN=451 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 217 | 29 | 10 |
| Malignant_lesions | 13 | 192 | 5 |
| body_of_stomach_normal | 1 | 4 | 200 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9394    0.8477    0.8912       256
     Malignant_lesions     0.8533    0.9143    0.8828       210
body_of_stomach_normal     0.9302    0.9756    0.9524       205

              accuracy                         0.9076       671
             macro avg     0.9077    0.9125    0.9088       671
          weighted avg     0.9097    0.9076    0.9072       671

```