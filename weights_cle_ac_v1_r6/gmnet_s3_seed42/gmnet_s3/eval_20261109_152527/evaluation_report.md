# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/gmnet_s3_seed42/gmnet_s3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9238 |
| Accuracy | 0.9236 |
| Macro-AUROC (OvR) | 0.9912 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9237 |
| Balanced Acc | 0.9213 |
| Macro-Sensitivity | 0.9213 |
| Macro-Specificity | 0.9609 |
| Macro-Precision | 0.9292 |
| AUC-PR (Macro) | 0.9836 |
| Cohen's Kappa | 0.8848 |
| MCC | 0.8862 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 865.8 (1.15 ms/张) |
| End-to-End FPS | 211.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9158 | 0.8634 | 0.9011 | 0.9656 | 0.9869 | 0.9780 | 208 |
| Malignant_lesions | 0.8580 | 0.9825 | 0.9557 | 0.9042 | 0.9402 | 0.9886 | 0.9763 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9844 | 0.9686 | 0.9661 | 0.9818 | 0.9982 | 0.9966 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=31 TN=337 FN=12
- **Malignant_lesions**: TP=151 FP=7 TN=393 FN=25
- **body_of_stomach_normal**: TP=185 FP=6 TN=378 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 7 | 5 |
| Malignant_lesions | 24 | 151 | 1 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8634    0.9423    0.9011       208
     Malignant_lesions     0.9557    0.8580    0.9042       176
body_of_stomach_normal     0.9686    0.9635    0.9661       192

              accuracy                         0.9236       576
             macro avg     0.9292    0.9213    0.9238       576
          weighted avg     0.9267    0.9236    0.9237       576

```