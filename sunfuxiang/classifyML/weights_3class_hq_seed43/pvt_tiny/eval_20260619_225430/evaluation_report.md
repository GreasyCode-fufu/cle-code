# pvt_tiny 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_hq_seed43/pvt_tiny/pvt_tiny_3class_hq_seed43_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9050 |
| Accuracy | 0.9031 |
| Macro-AUROC (OvR) | 0.9786 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9031 |
| Balanced Acc | 0.9068 |
| Macro-Sensitivity | 0.9068 |
| Macro-Specificity | 0.9513 |
| Macro-Precision | 0.9042 |
| AUC-PR (Macro) | 0.9622 |
| Cohen's Kappa | 0.8542 |
| MCC | 0.8547 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2899.0 (0.34 ms/张) |
| End-to-End FPS | 222.7 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8594 | 0.9470 | 0.9091 | 0.8835 | 0.9161 | 0.9718 | 0.9595 | 256 |
| Malignant_lesions | 0.8952 | 0.9262 | 0.8468 | 0.8704 | 0.9510 | 0.9682 | 0.9372 | 210 |
| body_of_stomach_normal | 0.9659 | 0.9807 | 0.9565 | 0.9612 | 0.9849 | 0.9958 | 0.9898 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=220 FP=22 TN=393 FN=36
- **Malignant_lesions**: TP=188 FP=34 TN=427 FN=22
- **body_of_stomach_normal**: TP=198 FP=9 TN=457 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 220 | 30 | 6 |
| Malignant_lesions | 19 | 188 | 3 |
| body_of_stomach_normal | 3 | 4 | 198 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9091    0.8594    0.8835       256
     Malignant_lesions     0.8468    0.8952    0.8704       210
body_of_stomach_normal     0.9565    0.9659    0.9612       205

              accuracy                         0.9031       671
             macro avg     0.9042    0.9068    0.9050       671
          weighted avg     0.9041    0.9031    0.9031       671

```