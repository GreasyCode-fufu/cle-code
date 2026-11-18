# convnext_v2abc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2abc_seed44/convnext_v2abc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9599 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9600 |
| Balanced Acc | 0.9594 |
| Macro-Sensitivity | 0.9594 |
| Macro-Specificity | 0.9798 |
| Macro-Precision | 0.9605 |
| AUC-PR (Macro) | 0.9927 |
| Cohen's Kappa | 0.9399 |
| MCC | 0.9400 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.66G |
| FLOPs | 9.31G |
| Params | 28.67M |
| Pure-Inference FPS | 527.6 (1.90 ms/张) |
| End-to-End FPS | 194.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9674 | 0.9431 | 0.9499 | 0.9753 | 0.9952 | 0.9918 | 208 |
| Malignant_lesions | 0.9318 | 0.9825 | 0.9591 | 0.9452 | 0.9704 | 0.9946 | 0.9879 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9896 | 0.9794 | 0.9845 | 0.9948 | 0.9990 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=12 TN=356 FN=9
- **Malignant_lesions**: TP=164 FP=7 TN=393 FN=12
- **body_of_stomach_normal**: TP=190 FP=4 TN=380 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 10 | 164 | 2 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9431    0.9567    0.9499       208
     Malignant_lesions     0.9591    0.9318    0.9452       176
body_of_stomach_normal     0.9794    0.9896    0.9845       192

              accuracy                         0.9601       576
             macro avg     0.9605    0.9594    0.9599       576
          weighted avg     0.9601    0.9601    0.9600       576

```