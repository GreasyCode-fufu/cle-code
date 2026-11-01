# convnext_cmp_fse 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_cmp_fse_seed43/convnext_cmp_fse/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9685 |
| Accuracy | 0.9688 |
| Macro-AUROC (OvR) | 0.9967 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9687 |
| Balanced Acc | 0.9684 |
| Macro-Sensitivity | 0.9684 |
| Macro-Specificity | 0.9843 |
| Macro-Precision | 0.9686 |
| AUC-PR (Macro) | 0.9935 |
| Cohen's Kappa | 0.9530 |
| MCC | 0.9530 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 512.0 (1.95 ms/张) |
| End-to-End FPS | 190.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9783 | 0.9615 | 0.9615 | 0.9783 | 0.9953 | 0.9923 | 208 |
| Malignant_lesions | 0.9489 | 0.9825 | 0.9598 | 0.9543 | 0.9776 | 0.9957 | 0.9901 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9922 | 0.9845 | 0.9896 | 0.9974 | 0.9990 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=8 TN=360 FN=8
- **Malignant_lesions**: TP=167 FP=7 TN=393 FN=9
- **body_of_stomach_normal**: TP=191 FP=3 TN=381 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 7 | 1 |
| Malignant_lesions | 7 | 167 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9615    0.9615    0.9615       208
     Malignant_lesions     0.9598    0.9489    0.9543       176
body_of_stomach_normal     0.9845    0.9948    0.9896       192

              accuracy                         0.9688       576
             macro avg     0.9686    0.9684    0.9685       576
          weighted avg     0.9687    0.9688    0.9687       576

```