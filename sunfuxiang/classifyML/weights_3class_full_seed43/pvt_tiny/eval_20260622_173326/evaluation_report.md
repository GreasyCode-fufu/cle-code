# pvt_tiny 评估报告

- 权重: `weights_3class_full_seed43/pvt_tiny/pvt_tiny_3class_full_seed43_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8581 |
| Accuracy | 0.8569 |
| Macro-AUROC (OvR) | 0.9580 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8557 |
| Balanced Acc | 0.8561 |
| Macro-Sensitivity | 0.8561 |
| Macro-Specificity | 0.9260 |
| Macro-Precision | 0.8650 |
| AUC-PR (Macro) | 0.9298 |
| Cohen's Kappa | 0.7833 |
| MCC | 0.7857 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2905.6 (0.34 ms/张) |
| End-to-End FPS | 250.6 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8789 | 0.8578 | 0.7923 | 0.8333 | 0.9199 | 0.9415 | 0.9168 | 256 |
| Malignant_lesions | 0.7333 | 0.9458 | 0.8603 | 0.7918 | 0.8862 | 0.9408 | 0.8906 | 210 |
| body_of_stomach_normal | 0.9561 | 0.9742 | 0.9423 | 0.9492 | 0.9806 | 0.9917 | 0.9819 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=225 FP=59 TN=356 FN=31
- **Malignant_lesions**: TP=154 FP=25 TN=436 FN=56
- **body_of_stomach_normal**: TP=196 FP=12 TN=454 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 225 | 23 | 8 |
| Malignant_lesions | 52 | 154 | 4 |
| body_of_stomach_normal | 7 | 2 | 196 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.7923    0.8789    0.8333       256
     Malignant_lesions     0.8603    0.7333    0.7918       210
body_of_stomach_normal     0.9423    0.9561    0.9492       205

              accuracy                         0.8569       671
             macro avg     0.8650    0.8561    0.8581       671
          weighted avg     0.8594    0.8569    0.8557       671

```