# convnext_wds 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_seed42/convnext_wds/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9532 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9950 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9532 |
| Balanced Acc | 0.9525 |
| Macro-Sensitivity | 0.9525 |
| Macro-Specificity | 0.9763 |
| Macro-Precision | 0.9542 |
| AUC-PR (Macro) | 0.9911 |
| Cohen's Kappa | 0.9295 |
| MCC | 0.9296 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.49G |
| FLOPs | 8.97G |
| Params | 27.81M |
| Pure-Inference FPS | 847.6 (1.18 ms/张) |
| End-to-End FPS | 227.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9592 | 0.9302 | 0.9456 | 0.9778 | 0.9925 | 0.9872 | 208 |
| Malignant_lesions | 0.9375 | 0.9800 | 0.9538 | 0.9456 | 0.9727 | 0.9940 | 0.9892 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9896 | 0.9787 | 0.9684 | 0.9794 | 0.9984 | 0.9969 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=15 TN=353 FN=8
- **Malignant_lesions**: TP=165 FP=8 TN=392 FN=11
- **body_of_stomach_normal**: TP=184 FP=4 TN=380 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 7 | 1 |
| Malignant_lesions | 8 | 165 | 3 |
| body_of_stomach_normal | 7 | 1 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9302    0.9615    0.9456       208
     Malignant_lesions     0.9538    0.9375    0.9456       176
body_of_stomach_normal     0.9787    0.9583    0.9684       192

              accuracy                         0.9531       576
             macro avg     0.9542    0.9525    0.9532       576
          weighted avg     0.9536    0.9531    0.9532       576

```