# resnet34_degconv_cpca_v3 评估报告

- 权重: `./weights/resnet34_degconv_cpca_v3/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9009 |
| Accuracy | 0.9001 |
| Macro-AUROC (OvR) | 0.9824 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8994 |
| Balanced Acc | 0.9058 |
| Macro-Sensitivity | 0.9058 |
| Macro-Specificity | 0.9507 |
| Macro-Precision | 0.8998 |
| AUC-PR (Macro) | 0.9684 |
| Cohen's Kappa | 0.8502 |
| MCC | 0.8521 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.76G |
| FLOPs | 7.53G |
| Params | 21.86M |
| Pure-Inference FPS | 3493.9 (0.29 ms/张) |
| End-to-End FPS | 235.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8320 | 0.9687 | 0.9425 | 0.8838 | 0.9034 | 0.9756 | 0.9666 | 256 |
| Malignant_lesions | 0.9048 | 0.9328 | 0.8597 | 0.8817 | 0.9556 | 0.9754 | 0.9460 | 210 |
| body_of_stomach_normal | 0.9805 | 0.9506 | 0.8973 | 0.9371 | 0.9911 | 0.9962 | 0.9926 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=213 FP=13 TN=402 FN=43
- **Malignant_lesions**: TP=190 FP=31 TN=430 FN=20
- **body_of_stomach_normal**: TP=201 FP=23 TN=443 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 213 | 29 | 14 |
| Malignant_lesions | 11 | 190 | 9 |
| body_of_stomach_normal | 2 | 2 | 201 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9425    0.8320    0.8838       256
     Malignant_lesions     0.8597    0.9048    0.8817       210
body_of_stomach_normal     0.8973    0.9805    0.9371       205

              accuracy                         0.9001       671
             macro avg     0.8998    0.9058    0.9009       671
          weighted avg     0.9028    0.9001    0.8994       671

```