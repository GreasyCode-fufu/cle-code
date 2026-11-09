# mbv3_crt 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_crt_seed45/mbv3_crt/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9375 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9903 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9376 |
| Balanced Acc | 0.9359 |
| Macro-Sensitivity | 0.9359 |
| Macro-Specificity | 0.9682 |
| Macro-Precision | 0.9404 |
| AUC-PR (Macro) | 0.9821 |
| Cohen's Kappa | 0.9059 |
| MCC | 0.9065 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 2465.1 (0.41 ms/张) |
| End-to-End FPS | 7.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9375 | 0.8964 | 0.9256 | 0.9746 | 0.9876 | 0.9789 | 208 |
| Malignant_lesions | 0.8977 | 0.9775 | 0.9461 | 0.9213 | 0.9560 | 0.9862 | 0.9723 | 176 |
| body_of_stomach_normal | 0.9531 | 0.9896 | 0.9786 | 0.9657 | 0.9769 | 0.9971 | 0.9950 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=23 TN=345 FN=9
- **Malignant_lesions**: TP=158 FP=9 TN=391 FN=18
- **body_of_stomach_normal**: TP=183 FP=4 TN=380 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 16 | 158 | 2 |
| body_of_stomach_normal | 7 | 2 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8964    0.9567    0.9256       208
     Malignant_lesions     0.9461    0.8977    0.9213       176
body_of_stomach_normal     0.9786    0.9531    0.9657       192

              accuracy                         0.9375       576
             macro avg     0.9404    0.9359    0.9375       576
          weighted avg     0.9390    0.9375    0.9376       576

```