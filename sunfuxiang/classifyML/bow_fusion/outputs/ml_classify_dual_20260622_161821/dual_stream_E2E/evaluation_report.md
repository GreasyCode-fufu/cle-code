# dual_stream-E2E 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8974 |
| Accuracy | 0.8957 |
| Macro-AUROC (OvR) | 0.9783 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8954 |
| Balanced Acc | 0.9028 |
| Macro-Sensitivity | 0.9028 |
| Macro-Specificity | 0.9487 |
| Macro-Precision | 0.8998 |
| AUC-PR (Macro) | 0.9624 |
| Cohen's Kappa | 0.8437 |
| MCC | 0.8477 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 373.55M |
| FLOPs | 747.10M |
| Params | 3.70M |
| Pure-Inference FPS | 822.4 (1.22 ms/张) |
| End-to-End FPS | 822.4 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8047 | 0.9759 | 0.9537 | 0.8729 | 0.8901 | 0.9677 | 0.9597 | 256 |
| Malignant_lesions | 0.9476 | 0.9002 | 0.8122 | 0.8747 | 0.9742 | 0.9722 | 0.9384 | 210 |
| body_of_stomach_normal | 0.9561 | 0.9700 | 0.9333 | 0.9446 | 0.9805 | 0.9952 | 0.9892 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=206 FP=10 TN=405 FN=50
- **Malignant_lesions**: TP=199 FP=46 TN=415 FN=11
- **body_of_stomach_normal**: TP=196 FP=14 TN=452 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 206 | 39 | 11 |
| Malignant_lesions | 8 | 199 | 3 |
| body_of_stomach_normal | 2 | 7 | 196 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9537    0.8047    0.8729       256
     Malignant_lesions     0.8122    0.9476    0.8747       210
body_of_stomach_normal     0.9333    0.9561    0.9446       205

              accuracy                         0.8957       671
             macro avg     0.8998    0.9028    0.8974       671
          weighted avg     0.9032    0.8957    0.8954       671

```