# convnext_fesr2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fesr2_seed44/convnext_fesr2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9730 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9986 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9790 |
| Balanced Acc | 0.9736 |
| Macro-Sensitivity | 0.9736 |
| Macro-Specificity | 0.9895 |
| Macro-Precision | 0.9724 |
| AUC-PR (Macro) | 0.9954 |
| Cohen's Kappa | 0.9665 |
| MCC | 0.9665 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1300.8 (0.77 ms/张) |
| End-to-End FPS | 249.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9828 | 0.9796 | 0.9821 | 0.9870 | 0.9986 | 0.9984 | 195 |
| Malignant_lesions | 0.9494 | 0.9856 | 0.9375 | 0.9434 | 0.9885 | 0.9972 | 0.9878 | 79 |
| body_of_stomach_normal | 0.9869 | 1.0000 | 1.0000 | 0.9934 | 0.9928 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=4 TN=228 FN=3
- **Malignant_lesions**: TP=75 FP=5 TN=343 FN=4
- **body_of_stomach_normal**: TP=151 FP=0 TN=274 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 3 | 0 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 2 | 151 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9796    0.9846    0.9821       195
     Malignant_lesions     0.9375    0.9494    0.9434        79
body_of_stomach_normal     1.0000    0.9869    0.9934       153

              accuracy                         0.9789       427
             macro avg     0.9724    0.9736    0.9730       427
          weighted avg     0.9791    0.9789    0.9790       427

```