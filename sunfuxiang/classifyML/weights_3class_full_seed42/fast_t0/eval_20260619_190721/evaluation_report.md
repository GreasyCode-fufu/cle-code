# fast_t0 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed42/fast_t0/fast_t0_3class_full_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8864 |
| Accuracy | 0.8852 |
| Macro-AUROC (OvR) | 0.9693 |

## 补充指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8846 |
| Balanced Acc | 0.8866 |
| Macro-Precision | 0.8870 |
| Macro-Recall | 0.8866 |
| AUC-PR (Macro) | 0.9430 |
| Cohen's Kappa | 0.8268 |
| MCC | 0.8271 |

## 每类指标

| 类 | Precision | Recall | F1 | Specificity | AUROC | support |
|---|---|---|---|---|---|---|
| Benign_lesions | 0.8582 | 0.8750 | 0.8665 | 0.9108 | 0.9539 | 256 |
| Malignant_lesions | 0.8687 | 0.8190 | 0.8431 | 0.9436 | 0.9587 | 210 |
| body_of_stomach_normal | 0.9340 | 0.9659 | 0.9496 | 0.9700 | 0.9953 | 205 |

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 224 | 23 | 9 |
| Malignant_lesions | 33 | 172 | 5 |
| body_of_stomach_normal | 4 | 3 | 198 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8582    0.8750    0.8665       256
     Malignant_lesions     0.8687    0.8190    0.8431       210
body_of_stomach_normal     0.9340    0.9659    0.9496       205

              accuracy                         0.8852       671
             macro avg     0.8870    0.8866    0.8864       671
          weighted avg     0.8846    0.8852    0.8846       671

```