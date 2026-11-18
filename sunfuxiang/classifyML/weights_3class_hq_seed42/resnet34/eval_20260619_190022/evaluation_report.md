# resnet34 评估报告

- 权重: `weights_3class_hq_seed42/resnet34/resnet34_3class_hq_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9044 |
| Accuracy | 0.9031 |
| Macro-AUROC (OvR) | 0.9812 |

## 补充指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9027 |
| Balanced Acc | 0.9091 |
| Macro-Precision | 0.9043 |
| Macro-Recall | 0.9091 |
| AUC-PR (Macro) | 0.9659 |
| Cohen's Kappa | 0.8547 |
| MCC | 0.8570 |

## 每类指标

| 类 | Precision | Recall | F1 | Specificity | AUROC | support |
|---|---|---|---|---|---|---|
| Benign_lesions | 0.9464 | 0.8281 | 0.8833 | 0.9711 | 0.9749 | 256 |
| Malignant_lesions | 0.8412 | 0.9333 | 0.8849 | 0.9197 | 0.9730 | 210 |
| body_of_stomach_normal | 0.9252 | 0.9659 | 0.9451 | 0.9657 | 0.9955 | 205 |

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 212 | 33 | 11 |
| Malignant_lesions | 9 | 196 | 5 |
| body_of_stomach_normal | 3 | 4 | 198 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9464    0.8281    0.8833       256
     Malignant_lesions     0.8412    0.9333    0.8849       210
body_of_stomach_normal     0.9252    0.9659    0.9451       205

              accuracy                         0.9031       671
             macro avg     0.9043    0.9091    0.9044       671
          weighted avg     0.9070    0.9031    0.9027       671

```