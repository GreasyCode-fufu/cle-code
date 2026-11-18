# dual_stream-E2E 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca-2/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8935 |
| Accuracy | 0.8927 |
| Macro-AUROC (OvR) | 0.9781 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8920 |
| Balanced Acc | 0.8955 |
| Macro-Sensitivity | 0.8955 |
| Macro-Specificity | 0.9459 |
| Macro-Precision | 0.8928 |
| AUC-PR (Macro) | 0.9626 |
| Cohen's Kappa | 0.8384 |
| MCC | 0.8390 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 372.86M |
| FLOPs | 745.72M |
| Params | 3.69M |
| Pure-Inference FPS | 811.0 (1.23 ms/张) |
| End-to-End FPS | 811.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8633 | 0.9325 | 0.8876 | 0.8752 | 0.9171 | 0.9672 | 0.9523 | 256 |
| Malignant_lesions | 0.8524 | 0.9523 | 0.8905 | 0.8710 | 0.9340 | 0.9716 | 0.9447 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9528 | 0.9005 | 0.9343 | 0.9867 | 0.9955 | 0.9909 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=221 FP=28 TN=387 FN=35
- **Malignant_lesions**: TP=179 FP=22 TN=439 FN=31
- **body_of_stomach_normal**: TP=199 FP=22 TN=444 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 221 | 19 | 16 |
| Malignant_lesions | 25 | 179 | 6 |
| body_of_stomach_normal | 3 | 3 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8876    0.8633    0.8752       256
     Malignant_lesions     0.8905    0.8524    0.8710       210
body_of_stomach_normal     0.9005    0.9707    0.9343       205

              accuracy                         0.8927       671
             macro avg     0.8928    0.8955    0.8935       671
          weighted avg     0.8924    0.8927    0.8920       671

```