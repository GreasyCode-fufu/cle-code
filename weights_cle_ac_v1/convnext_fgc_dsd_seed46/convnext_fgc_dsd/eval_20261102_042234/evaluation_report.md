# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_dsd_seed46/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9475 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9477 |
| Balanced Acc | 0.9480 |
| Macro-Sensitivity | 0.9480 |
| Macro-Specificity | 0.9740 |
| Macro-Precision | 0.9473 |
| AUC-PR (Macro) | 0.9907 |
| Cohen's Kappa | 0.9217 |
| MCC | 0.9218 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 1503.5 (0.67 ms/张) |
| End-to-End FPS | 266.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9701 | 0.9458 | 0.9343 | 0.9571 | 0.9916 | 0.9876 | 208 |
| Malignant_lesions | 0.9261 | 0.9700 | 0.9314 | 0.9288 | 0.9676 | 0.9937 | 0.9860 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9818 | 0.9646 | 0.9795 | 0.9974 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=11 TN=357 FN=16
- **Malignant_lesions**: TP=163 FP=12 TN=388 FN=13
- **body_of_stomach_normal**: TP=191 FP=7 TN=377 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 12 | 4 |
| Malignant_lesions | 10 | 163 | 3 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9458    0.9231    0.9343       208
     Malignant_lesions     0.9314    0.9261    0.9288       176
body_of_stomach_normal     0.9646    0.9948    0.9795       192

              accuracy                         0.9479       576
             macro avg     0.9473    0.9480    0.9475       576
          weighted avg     0.9477    0.9479    0.9477       576

```