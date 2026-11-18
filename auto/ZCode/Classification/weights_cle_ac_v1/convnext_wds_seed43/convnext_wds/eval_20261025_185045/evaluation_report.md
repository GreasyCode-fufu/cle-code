# convnext_wds 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_seed43/convnext_wds/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9653 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9964 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9653 |
| Balanced Acc | 0.9657 |
| Macro-Sensitivity | 0.9657 |
| Macro-Specificity | 0.9827 |
| Macro-Precision | 0.9650 |
| AUC-PR (Macro) | 0.9932 |
| Cohen's Kappa | 0.9478 |
| MCC | 0.9479 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.49G |
| FLOPs | 8.97G |
| Params | 27.81M |
| Pure-Inference FPS | 1697.4 (0.59 ms/张) |
| End-to-End FPS | 269.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9783 | 0.9612 | 0.9565 | 0.9730 | 0.9958 | 0.9927 | 208 |
| Malignant_lesions | 0.9659 | 0.9750 | 0.9444 | 0.9551 | 0.9848 | 0.9941 | 0.9883 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9948 | 0.9895 | 0.9843 | 0.9896 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=8 TN=360 FN=10
- **Malignant_lesions**: TP=170 FP=10 TN=390 FN=6
- **body_of_stomach_normal**: TP=188 FP=2 TN=382 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 9 | 1 |
| Malignant_lesions | 5 | 170 | 1 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9612    0.9519    0.9565       208
     Malignant_lesions     0.9444    0.9659    0.9551       176
body_of_stomach_normal     0.9895    0.9792    0.9843       192

              accuracy                         0.9653       576
             macro avg     0.9650    0.9657    0.9653       576
          weighted avg     0.9655    0.9653    0.9653       576

```