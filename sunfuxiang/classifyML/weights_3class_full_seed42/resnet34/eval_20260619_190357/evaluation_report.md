# resnet34 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed42/resnet34/resnet34_3class_full_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8781 |
| Accuracy | 0.8763 |
| Macro-AUROC (OvR) | 0.9643 |

## 补充指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8761 |
| Balanced Acc | 0.8785 |
| Macro-Precision | 0.8778 |
| Macro-Recall | 0.8785 |
| AUC-PR (Macro) | 0.9318 |
| Cohen's Kappa | 0.8135 |
| MCC | 0.8135 |

## 每类指标

| 类 | Precision | Recall | F1 | Specificity | AUROC | support |
|---|---|---|---|---|---|---|
| Benign_lesions | 0.8588 | 0.8555 | 0.8571 | 0.9133 | 0.9506 | 256 |
| Malignant_lesions | 0.8278 | 0.8238 | 0.8258 | 0.9219 | 0.9480 | 210 |
| body_of_stomach_normal | 0.9469 | 0.9561 | 0.9515 | 0.9764 | 0.9944 | 205 |

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 219 | 29 | 8 |
| Malignant_lesions | 34 | 173 | 3 |
| body_of_stomach_normal | 2 | 7 | 196 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8588    0.8555    0.8571       256
     Malignant_lesions     0.8278    0.8238    0.8258       210
body_of_stomach_normal     0.9469    0.9561    0.9515       205

              accuracy                         0.8763       671
             macro avg     0.8778    0.8785    0.8781       671
          weighted avg     0.8760    0.8763    0.8761       671

```