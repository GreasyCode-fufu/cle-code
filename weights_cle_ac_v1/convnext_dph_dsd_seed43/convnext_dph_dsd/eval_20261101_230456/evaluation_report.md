# convnext_dph_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_dsd_seed43/convnext_dph_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9632 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9965 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9634 |
| Balanced Acc | 0.9620 |
| Macro-Sensitivity | 0.9620 |
| Macro-Specificity | 0.9814 |
| Macro-Precision | 0.9653 |
| AUC-PR (Macro) | 0.9934 |
| Cohen's Kappa | 0.9451 |
| MCC | 0.9455 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 28.68M |
| Pure-Inference FPS | 960.5 (1.04 ms/张) |
| End-to-End FPS | 235.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9620 | 0.9355 | 0.9553 | 0.9861 | 0.9954 | 0.9924 | 208 |
| Malignant_lesions | 0.9205 | 0.9900 | 0.9759 | 0.9474 | 0.9659 | 0.9946 | 0.9888 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9922 | 0.9845 | 0.9870 | 0.9948 | 0.9995 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=14 TN=354 FN=5
- **Malignant_lesions**: TP=162 FP=4 TN=396 FN=14
- **body_of_stomach_normal**: TP=190 FP=3 TN=381 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 4 | 1 |
| Malignant_lesions | 12 | 162 | 2 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9355    0.9760    0.9553       208
     Malignant_lesions     0.9759    0.9205    0.9474       176
body_of_stomach_normal     0.9845    0.9896    0.9870       192

              accuracy                         0.9635       576
             macro avg     0.9653    0.9620    0.9632       576
          weighted avg     0.9642    0.9635    0.9634       576

```