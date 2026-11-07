# convnext_v2ccw 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2ccw_seed42/convnext_v2ccw/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9741 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9980 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9788 |
| Balanced Acc | 0.9721 |
| Macro-Sensitivity | 0.9721 |
| Macro-Specificity | 0.9887 |
| Macro-Precision | 0.9763 |
| AUC-PR (Macro) | 0.9940 |
| Cohen's Kappa | 0.9664 |
| MCC | 0.9665 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 437.3 (2.29 ms/张) |
| End-to-End FPS | 181.1 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9784 | 0.9745 | 0.9770 | 0.9827 | 0.9977 | 0.9973 | 195 |
| Malignant_lesions | 0.9367 | 0.9914 | 0.9610 | 0.9487 | 0.9857 | 0.9963 | 0.9848 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=5 TN=227 FN=4
- **Malignant_lesions**: TP=74 FP=3 TN=345 FN=5
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 3 | 1 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9745    0.9795    0.9770       195
     Malignant_lesions     0.9610    0.9367    0.9487        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9789       427
             macro avg     0.9763    0.9721    0.9741       427
          weighted avg     0.9788    0.9789    0.9788       427

```