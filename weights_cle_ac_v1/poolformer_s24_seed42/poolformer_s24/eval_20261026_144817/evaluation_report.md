# poolformer_s24 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/poolformer_s24_seed42/poolformer_s24/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9484 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9483 |
| Balanced Acc | 0.9453 |
| Macro-Sensitivity | 0.9453 |
| Macro-Specificity | 0.9730 |
| Macro-Precision | 0.9550 |
| AUC-PR (Macro) | 0.9931 |
| Cohen's Kappa | 0.9215 |
| MCC | 0.9234 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.39G |
| FLOPs | 6.79G |
| Params | 20.84M |
| Pure-Inference FPS | 821.8 (1.22 ms/张) |
| End-to-End FPS | 216.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9856 | 0.9266 | 0.8836 | 0.9318 | 0.9913 | 0.9946 | 0.9911 | 208 |
| Malignant_lesions | 0.8920 | 0.9925 | 0.9812 | 0.9345 | 0.9543 | 0.9951 | 0.9897 | 176 |
| body_of_stomach_normal | 0.9583 | 1.0000 | 1.0000 | 0.9787 | 0.9796 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=205 FP=27 TN=341 FN=3
- **Malignant_lesions**: TP=157 FP=3 TN=397 FN=19
- **body_of_stomach_normal**: TP=184 FP=0 TN=384 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 205 | 3 | 0 |
| Malignant_lesions | 19 | 157 | 0 |
| body_of_stomach_normal | 8 | 0 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8836    0.9856    0.9318       208
     Malignant_lesions     0.9812    0.8920    0.9345       176
body_of_stomach_normal     1.0000    0.9583    0.9787       192

              accuracy                         0.9479       576
             macro avg     0.9550    0.9453    0.9484       576
          weighted avg     0.9522    0.9479    0.9483       576

```