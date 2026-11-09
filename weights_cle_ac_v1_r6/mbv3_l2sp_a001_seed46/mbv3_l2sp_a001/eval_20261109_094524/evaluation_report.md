# mbv3_l2sp_a001 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_l2sp_a001_seed46/mbv3_l2sp_a001/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9356 |
| Accuracy | 0.9358 |
| Macro-AUROC (OvR) | 0.9938 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9359 |
| Balanced Acc | 0.9329 |
| Macro-Sensitivity | 0.9329 |
| Macro-Specificity | 0.9670 |
| Macro-Precision | 0.9418 |
| AUC-PR (Macro) | 0.9885 |
| Cohen's Kappa | 0.9031 |
| MCC | 0.9048 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4414.2 (0.23 ms/张) |
| End-to-End FPS | 221.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9212 | 0.8739 | 0.9178 | 0.9798 | 0.9902 | 0.9840 | 208 |
| Malignant_lesions | 0.8636 | 0.9850 | 0.9620 | 0.9102 | 0.9426 | 0.9922 | 0.9834 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9948 | 0.9894 | 0.9789 | 0.9845 | 0.9991 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=29 TN=339 FN=7
- **Malignant_lesions**: TP=152 FP=6 TN=394 FN=24
- **body_of_stomach_normal**: TP=186 FP=2 TN=382 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 24 | 152 | 0 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8739    0.9663    0.9178       208
     Malignant_lesions     0.9620    0.8636    0.9102       176
body_of_stomach_normal     0.9894    0.9688    0.9789       192

              accuracy                         0.9358       576
             macro avg     0.9418    0.9329    0.9356       576
          weighted avg     0.9393    0.9358    0.9359       576

```