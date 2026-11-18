# convnext_dph4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph4_seed44/convnext_dph4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9651 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9957 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9652 |
| Balanced Acc | 0.9651 |
| Macro-Sensitivity | 0.9651 |
| Macro-Specificity | 0.9826 |
| Macro-Precision | 0.9652 |
| AUC-PR (Macro) | 0.9922 |
| Cohen's Kappa | 0.9478 |
| MCC | 0.9478 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 448.9 (2.23 ms/张) |
| End-to-End FPS | 183.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9755 | 0.9567 | 0.9567 | 0.9755 | 0.9937 | 0.9911 | 208 |
| Malignant_lesions | 0.9489 | 0.9800 | 0.9543 | 0.9516 | 0.9776 | 0.9944 | 0.9875 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9922 | 0.9845 | 0.9870 | 0.9948 | 0.9990 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=9 TN=359 FN=9
- **Malignant_lesions**: TP=167 FP=8 TN=392 FN=9
- **body_of_stomach_normal**: TP=190 FP=3 TN=381 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 7 | 167 | 2 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9567    0.9567    0.9567       208
     Malignant_lesions     0.9543    0.9489    0.9516       176
body_of_stomach_normal     0.9845    0.9896    0.9870       192

              accuracy                         0.9653       576
             macro avg     0.9652    0.9651    0.9651       576
          weighted avg     0.9652    0.9653    0.9652       576

```