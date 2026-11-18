# resnet34 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_hq_seed42/resnet34/resnet34_3class_hq_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9044 |
| Accuracy | 0.9031 |
| Macro-AUROC (OvR) | 0.9812 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9027 |
| Balanced Acc | 0.9091 |
| Macro-Sensitivity | 0.9091 |
| Macro-Specificity | 0.9522 |
| Macro-Precision | 0.9043 |
| AUC-PR (Macro) | 0.9659 |
| Cohen's Kappa | 0.8547 |
| MCC | 0.8570 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3801.2 (0.26 ms/张) |
| End-to-End FPS | 219.7 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8281 | 0.9711 | 0.9464 | 0.8833 | 0.9016 | 0.9749 | 0.9660 | 256 |
| Malignant_lesions | 0.9333 | 0.9197 | 0.8412 | 0.8849 | 0.9680 | 0.9730 | 0.9424 | 210 |
| body_of_stomach_normal | 0.9659 | 0.9657 | 0.9252 | 0.9451 | 0.9847 | 0.9955 | 0.9892 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=212 FP=12 TN=403 FN=44
- **Malignant_lesions**: TP=196 FP=37 TN=424 FN=14
- **body_of_stomach_normal**: TP=198 FP=16 TN=450 FN=7

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