# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_egb_scg_seed44/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9542 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9956 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9545 |
| Balanced Acc | 0.9537 |
| Macro-Sensitivity | 0.9537 |
| Macro-Specificity | 0.9772 |
| Macro-Precision | 0.9558 |
| AUC-PR (Macro) | 0.9919 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9326 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 313.9 (3.19 ms/张) |
| End-to-End FPS | 158.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9728 | 0.9519 | 0.9519 | 0.9728 | 0.9939 | 0.9886 | 208 |
| Malignant_lesions | 0.9091 | 0.9875 | 0.9697 | 0.9384 | 0.9611 | 0.9928 | 0.9872 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9714 | 0.9458 | 0.9722 | 1.0000 | 0.9999 | 0.9998 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=10 TN=358 FN=10
- **Malignant_lesions**: TP=160 FP=5 TN=395 FN=16
- **body_of_stomach_normal**: TP=192 FP=11 TN=373 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 5 | 5 |
| Malignant_lesions | 10 | 160 | 6 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9519    0.9519    0.9519       208
     Malignant_lesions     0.9697    0.9091    0.9384       176
body_of_stomach_normal     0.9458    1.0000    0.9722       192

              accuracy                         0.9549       576
             macro avg     0.9558    0.9537    0.9542       576
          weighted avg     0.9553    0.9549    0.9545       576

```