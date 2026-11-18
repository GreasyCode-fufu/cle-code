# mbv3_l2sp_a01 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_l2sp_a01_seed46/mbv3_l2sp_a01/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9075 |
| Accuracy | 0.9062 |
| Macro-AUROC (OvR) | 0.9895 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9072 |
| Balanced Acc | 0.9034 |
| Macro-Sensitivity | 0.9034 |
| Macro-Specificity | 0.9518 |
| Macro-Precision | 0.9187 |
| AUC-PR (Macro) | 0.9813 |
| Cohen's Kappa | 0.8585 |
| MCC | 0.8624 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4479.5 (0.22 ms/张) |
| End-to-End FPS | 165.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.8832 | 0.8238 | 0.8894 | 0.9789 | 0.9852 | 0.9757 | 208 |
| Malignant_lesions | 0.8636 | 0.9750 | 0.9383 | 0.8994 | 0.9420 | 0.9851 | 0.9714 | 176 |
| body_of_stomach_normal | 0.8802 | 0.9974 | 0.9941 | 0.9337 | 0.9433 | 0.9983 | 0.9967 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=43 TN=325 FN=7
- **Malignant_lesions**: TP=152 FP=10 TN=390 FN=24
- **body_of_stomach_normal**: TP=169 FP=1 TN=383 FN=23

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 24 | 152 | 0 |
| body_of_stomach_normal | 19 | 4 | 169 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8238    0.9663    0.8894       208
     Malignant_lesions     0.9383    0.8636    0.8994       176
body_of_stomach_normal     0.9941    0.8802    0.9337       192

              accuracy                         0.9062       576
             macro avg     0.9187    0.9034    0.9075       576
          weighted avg     0.9155    0.9062    0.9072       576

```