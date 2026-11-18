# convnext_fse_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fse_dsd_seed42/convnext_fse_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9511 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9926 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9515 |
| Balanced Acc | 0.9517 |
| Macro-Sensitivity | 0.9517 |
| Macro-Specificity | 0.9758 |
| Macro-Precision | 0.9508 |
| AUC-PR (Macro) | 0.9856 |
| Cohen's Kappa | 0.9270 |
| MCC | 0.9271 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 27.78M |
| Pure-Inference FPS | 356.7 (2.80 ms/张) |
| End-to-End FPS | 166.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9728 | 0.9510 | 0.9417 | 0.9624 | 0.9898 | 0.9796 | 208 |
| Malignant_lesions | 0.9432 | 0.9625 | 0.9171 | 0.9300 | 0.9747 | 0.9884 | 0.9781 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=10 TN=358 FN=14
- **Malignant_lesions**: TP=166 FP=15 TN=385 FN=10
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 13 | 1 |
| Malignant_lesions | 8 | 166 | 2 |
| body_of_stomach_normal | 2 | 2 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9510    0.9327    0.9417       208
     Malignant_lesions     0.9171    0.9432    0.9300       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9514       576
             macro avg     0.9508    0.9517    0.9511       576
          weighted avg     0.9517    0.9514    0.9515       576

```