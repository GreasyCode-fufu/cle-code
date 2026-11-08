# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz16_seed46/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9651 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9948 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9653 |
| Balanced Acc | 0.9651 |
| Macro-Sensitivity | 0.9651 |
| Macro-Specificity | 0.9826 |
| Macro-Precision | 0.9652 |
| AUC-PR (Macro) | 0.9895 |
| Cohen's Kappa | 0.9478 |
| MCC | 0.9478 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 5623.8 (0.18 ms/张) |
| End-to-End FPS | 178.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9783 | 0.9615 | 0.9615 | 0.9783 | 0.9916 | 0.9886 | 208 |
| Malignant_lesions | 0.9545 | 0.9825 | 0.9600 | 0.9573 | 0.9800 | 0.9948 | 0.9839 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9870 | 0.9741 | 0.9766 | 0.9896 | 0.9979 | 0.9961 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=8 TN=360 FN=8
- **Malignant_lesions**: TP=168 FP=7 TN=393 FN=8
- **body_of_stomach_normal**: TP=188 FP=5 TN=379 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 5 | 168 | 3 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9615    0.9615    0.9615       208
     Malignant_lesions     0.9600    0.9545    0.9573       176
body_of_stomach_normal     0.9741    0.9792    0.9766       192

              accuracy                         0.9653       576
             macro avg     0.9652    0.9651    0.9651       576
          weighted avg     0.9653    0.9653    0.9653       576

```