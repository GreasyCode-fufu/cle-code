# convnext_v2abc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2abc_seed43/convnext_v2abc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9730 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9984 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9788 |
| Balanced Acc | 0.9696 |
| Macro-Sensitivity | 0.9696 |
| Macro-Specificity | 0.9885 |
| Macro-Precision | 0.9767 |
| AUC-PR (Macro) | 0.9956 |
| Cohen's Kappa | 0.9664 |
| MCC | 0.9665 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.66G |
| FLOPs | 9.31G |
| Params | 28.67M |
| Pure-Inference FPS | 555.2 (1.80 ms/张) |
| End-to-End FPS | 195.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9741 | 0.9697 | 0.9771 | 0.9869 | 0.9982 | 0.9978 | 195 |
| Malignant_lesions | 0.9241 | 0.9914 | 0.9605 | 0.9419 | 0.9829 | 0.9969 | 0.9889 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=6 TN=226 FN=3
- **Malignant_lesions**: TP=73 FP=3 TN=345 FN=6
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 3 | 0 |
| Malignant_lesions | 6 | 73 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9697    0.9846    0.9771       195
     Malignant_lesions     0.9605    0.9241    0.9419        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9789       427
             macro avg     0.9767    0.9696    0.9730       427
          weighted avg     0.9789    0.9789    0.9788       427

```