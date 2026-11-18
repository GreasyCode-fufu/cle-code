# pvt_tiny 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_hq_seed42/pvt_tiny/pvt_tiny_3class_hq_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8958 |
| Accuracy | 0.8942 |
| Macro-AUROC (OvR) | 0.9756 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8935 |
| Balanced Acc | 0.9019 |
| Macro-Sensitivity | 0.9019 |
| Macro-Specificity | 0.9480 |
| Macro-Precision | 0.8978 |
| AUC-PR (Macro) | 0.9545 |
| Cohen's Kappa | 0.8415 |
| MCC | 0.8456 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2913.3 (0.34 ms/张) |
| End-to-End FPS | 247.1 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7969 | 0.9759 | 0.9533 | 0.8681 | 0.8862 | 0.9630 | 0.9566 | 256 |
| Malignant_lesions | 0.9429 | 0.9024 | 0.8148 | 0.8742 | 0.9720 | 0.9689 | 0.9180 | 210 |
| body_of_stomach_normal | 0.9659 | 0.9657 | 0.9252 | 0.9451 | 0.9847 | 0.9949 | 0.9888 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=10 TN=405 FN=52
- **Malignant_lesions**: TP=198 FP=45 TN=416 FN=12
- **body_of_stomach_normal**: TP=198 FP=16 TN=450 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 41 | 11 |
| Malignant_lesions | 7 | 198 | 5 |
| body_of_stomach_normal | 3 | 4 | 198 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9533    0.7969    0.8681       256
     Malignant_lesions     0.8148    0.9429    0.8742       210
body_of_stomach_normal     0.9252    0.9659    0.9451       205

              accuracy                         0.8942       671
             macro avg     0.8978    0.9019    0.8958       671
          weighted avg     0.9014    0.8942    0.8935       671

```