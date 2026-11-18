# fast_t0 评估报告

- 权重: `weights_3class_full_seed43/fast_t0/fast_t0_3class_full_seed43_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8690 |
| Accuracy | 0.8689 |
| Macro-AUROC (OvR) | 0.9621 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8672 |
| Balanced Acc | 0.8710 |
| Macro-Sensitivity | 0.8710 |
| Macro-Specificity | 0.9335 |
| Macro-Precision | 0.8689 |
| AUC-PR (Macro) | 0.9332 |
| Cohen's Kappa | 0.8023 |
| MCC | 0.8031 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 9086.1 (0.11 ms/张) |
| End-to-End FPS | 228.2 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8516 | 0.9084 | 0.8516 | 0.8516 | 0.9084 | 0.9472 | 0.9267 | 256 |
| Malignant_lesions | 0.7810 | 0.9371 | 0.8497 | 0.8139 | 0.9038 | 0.9455 | 0.8859 | 210 |
| body_of_stomach_normal | 0.9805 | 0.9549 | 0.9054 | 0.9415 | 0.9911 | 0.9937 | 0.9871 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=218 FP=38 TN=377 FN=38
- **Malignant_lesions**: TP=164 FP=29 TN=432 FN=46
- **body_of_stomach_normal**: TP=201 FP=21 TN=445 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 218 | 25 | 13 |
| Malignant_lesions | 38 | 164 | 8 |
| body_of_stomach_normal | 0 | 4 | 201 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8516    0.8516    0.8516       256
     Malignant_lesions     0.8497    0.7810    0.8139       210
body_of_stomach_normal     0.9054    0.9805    0.9415       205

              accuracy                         0.8689       671
             macro avg     0.8689    0.8710    0.8690       671
          weighted avg     0.8674    0.8689    0.8672       671

```