# resnet34_degconv_cpca_v3 评估报告

- 权重: `./weights/resnet34_degconv_cpca_v3-1/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9070 |
| Accuracy | 0.9061 |
| Macro-AUROC (OvR) | 0.9816 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9057 |
| Balanced Acc | 0.9115 |
| Macro-Sensitivity | 0.9115 |
| Macro-Specificity | 0.9537 |
| Macro-Precision | 0.9064 |
| AUC-PR (Macro) | 0.9655 |
| Cohen's Kappa | 0.8591 |
| MCC | 0.8611 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.76G |
| FLOPs | 7.53G |
| Params | 21.86M |
| Pure-Inference FPS | 3533.4 (0.28 ms/张) |
| End-to-End FPS | 231.8 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8398 | 0.9735 | 0.9513 | 0.8921 | 0.9079 | 0.9745 | 0.9662 | 256 |
| Malignant_lesions | 0.9238 | 0.9262 | 0.8509 | 0.8858 | 0.9639 | 0.9735 | 0.9379 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9614 | 0.9171 | 0.9431 | 0.9868 | 0.9967 | 0.9924 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=215 FP=11 TN=404 FN=41
- **Malignant_lesions**: TP=194 FP=34 TN=427 FN=16
- **body_of_stomach_normal**: TP=199 FP=18 TN=448 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 215 | 29 | 12 |
| Malignant_lesions | 10 | 194 | 6 |
| body_of_stomach_normal | 1 | 5 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9513    0.8398    0.8921       256
     Malignant_lesions     0.8509    0.9238    0.8858       210
body_of_stomach_normal     0.9171    0.9707    0.9431       205

              accuracy                         0.9061       671
             macro avg     0.9064    0.9115    0.9070       671
          weighted avg     0.9094    0.9061    0.9057       671

```