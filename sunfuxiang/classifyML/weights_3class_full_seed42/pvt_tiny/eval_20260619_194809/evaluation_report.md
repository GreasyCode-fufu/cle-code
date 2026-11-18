# pvt_tiny 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed42/pvt_tiny/pvt_tiny_3class_full_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8754 |
| Accuracy | 0.8733 |
| Macro-AUROC (OvR) | 0.9726 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8728 |
| Balanced Acc | 0.8816 |
| Macro-Sensitivity | 0.8816 |
| Macro-Specificity | 0.9376 |
| Macro-Precision | 0.8787 |
| AUC-PR (Macro) | 0.9528 |
| Cohen's Kappa | 0.8104 |
| MCC | 0.8152 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2915.2 (0.34 ms/张) |
| End-to-End FPS | 231.1 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7695 | 0.9687 | 0.9381 | 0.8455 | 0.8720 | 0.9639 | 0.9489 | 256 |
| Malignant_lesions | 0.9143 | 0.8807 | 0.7773 | 0.8403 | 0.9575 | 0.9603 | 0.9260 | 210 |
| body_of_stomach_normal | 0.9610 | 0.9635 | 0.9206 | 0.9403 | 0.9825 | 0.9935 | 0.9837 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=13 TN=402 FN=59
- **Malignant_lesions**: TP=192 FP=55 TN=406 FN=18
- **body_of_stomach_normal**: TP=197 FP=17 TN=449 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 50 | 9 |
| Malignant_lesions | 10 | 192 | 8 |
| body_of_stomach_normal | 3 | 5 | 197 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9381    0.7695    0.8455       256
     Malignant_lesions     0.7773    0.9143    0.8403       210
body_of_stomach_normal     0.9206    0.9610    0.9403       205

              accuracy                         0.8733       671
             macro avg     0.8787    0.8816    0.8754       671
          weighted avg     0.8824    0.8733    0.8728       671

```