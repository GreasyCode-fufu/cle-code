# repvit_m1 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed43/repvit_m1/repvit_m1_3class_full_seed43_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7760 |
| Accuracy | 0.7765 |
| Macro-AUROC (OvR) | 0.9213 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7739 |
| Balanced Acc | 0.7790 |
| Macro-Sensitivity | 0.7790 |
| Macro-Specificity | 0.8869 |
| Macro-Precision | 0.7744 |
| AUC-PR (Macro) | 0.8531 |
| Cohen's Kappa | 0.6630 |
| MCC | 0.6636 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 3830.8 (0.26 ms/张) |
| End-to-End FPS | 219.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7578 | 0.8530 | 0.7608 | 0.7593 | 0.8510 | 0.9003 | 0.8619 | 256 |
| Malignant_lesions | 0.6524 | 0.8742 | 0.7026 | 0.6765 | 0.8466 | 0.8825 | 0.7371 | 210 |
| body_of_stomach_normal | 0.9268 | 0.9335 | 0.8597 | 0.8920 | 0.9667 | 0.9812 | 0.9605 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=61 TN=354 FN=62
- **Malignant_lesions**: TP=137 FP=58 TN=403 FN=73
- **body_of_stomach_normal**: TP=190 FP=31 TN=435 FN=15

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 46 | 16 |
| Malignant_lesions | 58 | 137 | 15 |
| body_of_stomach_normal | 3 | 12 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.7608    0.7578    0.7593       256
     Malignant_lesions     0.7026    0.6524    0.6765       210
body_of_stomach_normal     0.8597    0.9268    0.8920       205

              accuracy                         0.7765       671
             macro avg     0.7744    0.7790    0.7760       671
          weighted avg     0.7728    0.7765    0.7739       671

```