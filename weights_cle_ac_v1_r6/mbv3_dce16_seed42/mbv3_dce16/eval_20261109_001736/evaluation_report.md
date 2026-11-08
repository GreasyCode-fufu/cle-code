# mbv3_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce16_seed42/mbv3_dce16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9551 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9942 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9549 |
| Balanced Acc | 0.9556 |
| Macro-Sensitivity | 0.9556 |
| Macro-Specificity | 0.9775 |
| Macro-Precision | 0.9548 |
| AUC-PR (Macro) | 0.9883 |
| Cohen's Kappa | 0.9322 |
| MCC | 0.9323 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 3932.3 (0.25 ms/张) |
| End-to-End FPS | 11.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9701 | 0.9463 | 0.9395 | 0.9623 | 0.9907 | 0.9848 | 208 |
| Malignant_lesions | 0.9602 | 0.9675 | 0.9286 | 0.9441 | 0.9822 | 0.9923 | 0.9810 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9948 | 0.9894 | 0.9816 | 0.9871 | 0.9995 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=11 TN=357 FN=14
- **Malignant_lesions**: TP=169 FP=13 TN=387 FN=7
- **body_of_stomach_normal**: TP=187 FP=2 TN=382 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 12 | 2 |
| Malignant_lesions | 7 | 169 | 0 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9463    0.9327    0.9395       208
     Malignant_lesions     0.9286    0.9602    0.9441       176
body_of_stomach_normal     0.9894    0.9740    0.9816       192

              accuracy                         0.9549       576
             macro avg     0.9548    0.9556    0.9551       576
          weighted avg     0.9553    0.9549    0.9549       576

```