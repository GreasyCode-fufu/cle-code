# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2c_seed43/convnext_v2c/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9669 |
| Accuracy | 0.9670 |
| Macro-AUROC (OvR) | 0.9950 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9670 |
| Balanced Acc | 0.9677 |
| Macro-Sensitivity | 0.9677 |
| Macro-Specificity | 0.9837 |
| Macro-Precision | 0.9664 |
| AUC-PR (Macro) | 0.9908 |
| Cohen's Kappa | 0.9504 |
| MCC | 0.9506 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1026.3 (0.97 ms/张) |
| End-to-End FPS | 231.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9864 | 0.9752 | 0.9610 | 0.9706 | 0.9925 | 0.9904 | 208 |
| Malignant_lesions | 0.9716 | 0.9750 | 0.9448 | 0.9580 | 0.9873 | 0.9936 | 0.9838 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9896 | 0.9793 | 0.9818 | 0.9922 | 0.9990 | 0.9980 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=5 TN=363 FN=11
- **Malignant_lesions**: TP=171 FP=10 TN=390 FN=5
- **body_of_stomach_normal**: TP=189 FP=4 TN=380 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 2 | 171 | 3 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9752    0.9471    0.9610       208
     Malignant_lesions     0.9448    0.9716    0.9580       176
body_of_stomach_normal     0.9793    0.9844    0.9818       192

              accuracy                         0.9670       576
             macro avg     0.9664    0.9677    0.9669       576
          weighted avg     0.9673    0.9670    0.9670       576

```