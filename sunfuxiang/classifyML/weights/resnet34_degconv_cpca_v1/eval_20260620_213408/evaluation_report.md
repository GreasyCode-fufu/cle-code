# resnet34_degconv_cpca_v1 评估报告

- 权重: `./weights/resnet34_degconv_cpca_v1/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9043 |
| Accuracy | 0.9031 |
| Macro-AUROC (OvR) | 0.9806 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9020 |
| Balanced Acc | 0.9106 |
| Macro-Sensitivity | 0.9106 |
| Macro-Specificity | 0.9523 |
| Macro-Precision | 0.9046 |
| AUC-PR (Macro) | 0.9634 |
| Cohen's Kappa | 0.8548 |
| MCC | 0.8582 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.86G |
| FLOPs | 7.73G |
| Params | 22.56M |
| Pure-Inference FPS | 982.3 (1.02 ms/张) |
| End-to-End FPS | 190.7 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8086 | 0.9759 | 0.9539 | 0.8753 | 0.8921 | 0.9732 | 0.9642 | 256 |
| Malignant_lesions | 0.9429 | 0.9219 | 0.8462 | 0.8919 | 0.9725 | 0.9731 | 0.9397 | 210 |
| body_of_stomach_normal | 0.9805 | 0.9592 | 0.9136 | 0.9459 | 0.9911 | 0.9954 | 0.9863 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=207 FP=10 TN=405 FN=49
- **Malignant_lesions**: TP=198 FP=36 TN=425 FN=12
- **body_of_stomach_normal**: TP=201 FP=19 TN=447 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 207 | 34 | 15 |
| Malignant_lesions | 8 | 198 | 4 |
| body_of_stomach_normal | 2 | 2 | 201 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9539    0.8086    0.8753       256
     Malignant_lesions     0.8462    0.9429    0.8919       210
body_of_stomach_normal     0.9136    0.9805    0.9459       205

              accuracy                         0.9031       671
             macro avg     0.9046    0.9106    0.9043       671
          weighted avg     0.9079    0.9031    0.9020       671

```