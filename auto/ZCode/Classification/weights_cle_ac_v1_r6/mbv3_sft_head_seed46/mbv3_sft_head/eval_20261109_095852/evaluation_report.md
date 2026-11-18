# mbv3_sft_head 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_head_seed46/mbv3_sft_head/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9389 |
| Accuracy | 0.9485 |
| Macro-AUROC (OvR) | 0.9914 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9492 |
| Balanced Acc | 0.9450 |
| Macro-Sensitivity | 0.9450 |
| Macro-Specificity | 0.9746 |
| Macro-Precision | 0.9341 |
| AUC-PR (Macro) | 0.9710 |
| Cohen's Kappa | 0.9185 |
| MCC | 0.9188 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3822.3 (0.26 ms/张) |
| End-to-End FPS | 94.3 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9436 | 0.9612 | 0.9534 | 0.9485 | 0.9530 | 0.9891 | 0.9891 | 195 |
| Malignant_lesions | 0.9241 | 0.9626 | 0.8488 | 0.8848 | 0.9824 | 0.9854 | 0.9242 | 79 |
| body_of_stomach_normal | 0.9673 | 1.0000 | 1.0000 | 0.9834 | 0.9821 | 0.9998 | 0.9996 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=184 FP=9 TN=223 FN=11
- **Malignant_lesions**: TP=73 FP=13 TN=335 FN=6
- **body_of_stomach_normal**: TP=148 FP=0 TN=274 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 184 | 11 | 0 |
| Malignant_lesions | 6 | 73 | 0 |
| body_of_stomach_normal | 3 | 2 | 148 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9534    0.9436    0.9485       195
     Malignant_lesions     0.8488    0.9241    0.8848        79
body_of_stomach_normal     1.0000    0.9673    0.9834       153

              accuracy                         0.9485       427
             macro avg     0.9341    0.9450    0.9389       427
          weighted avg     0.9507    0.9485    0.9492       427

```