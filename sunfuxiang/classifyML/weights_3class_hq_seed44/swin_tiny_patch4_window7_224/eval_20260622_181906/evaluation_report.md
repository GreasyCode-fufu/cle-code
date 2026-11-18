# swin_tiny_patch4_window7_224 评估报告

- 权重: `weights_3class_hq_seed44/swin_tiny_patch4_window7_224/swin_tiny_patch4_window7_224_3class_hq_seed44_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8982 |
| Accuracy | 0.8972 |
| Macro-AUROC (OvR) | 0.9749 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8975 |
| Balanced Acc | 0.8982 |
| Macro-Sensitivity | 0.8982 |
| Macro-Specificity | 0.9481 |
| Macro-Precision | 0.8985 |
| AUC-PR (Macro) | 0.9557 |
| Cohen's Kappa | 0.8450 |
| MCC | 0.8451 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.92G |
| Params | 27.94M |
| Pure-Inference FPS | 1526.8 (0.65 ms/张) |
| End-to-End FPS | 213.4 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8867 | 0.9373 | 0.8972 | 0.8919 | 0.9306 | 0.9690 | 0.9548 | 256 |
| Malignant_lesions | 0.8762 | 0.9284 | 0.8479 | 0.8618 | 0.9427 | 0.9644 | 0.9298 | 210 |
| body_of_stomach_normal | 0.9317 | 0.9785 | 0.9502 | 0.9409 | 0.9702 | 0.9913 | 0.9826 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=227 FP=26 TN=389 FN=29
- **Malignant_lesions**: TP=184 FP=33 TN=428 FN=26
- **body_of_stomach_normal**: TP=191 FP=10 TN=456 FN=14

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 227 | 26 | 3 |
| Malignant_lesions | 19 | 184 | 7 |
| body_of_stomach_normal | 7 | 7 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8972    0.8867    0.8919       256
     Malignant_lesions     0.8479    0.8762    0.8618       210
body_of_stomach_normal     0.9502    0.9317    0.9409       205

              accuracy                         0.8972       671
             macro avg     0.8985    0.8982    0.8982       671
          weighted avg     0.8980    0.8972    0.8975       671

```