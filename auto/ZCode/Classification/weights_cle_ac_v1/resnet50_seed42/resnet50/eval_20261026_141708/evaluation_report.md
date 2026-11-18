# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/resnet50_seed42/resnet50/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9507 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9942 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9511 |
| Balanced Acc | 0.9495 |
| Macro-Sensitivity | 0.9495 |
| Macro-Specificity | 0.9752 |
| Macro-Precision | 0.9534 |
| AUC-PR (Macro) | 0.9876 |
| Cohen's Kappa | 0.9268 |
| MCC | 0.9274 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 939.6 (1.06 ms/张) |
| End-to-End FPS | 212.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9538 | 0.9217 | 0.9412 | 0.9777 | 0.9913 | 0.9859 | 208 |
| Malignant_lesions | 0.8920 | 0.9875 | 0.9691 | 0.9290 | 0.9541 | 0.9916 | 0.9775 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9844 | 0.9695 | 0.9820 | 0.9974 | 0.9997 | 0.9995 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=17 TN=351 FN=8
- **Malignant_lesions**: TP=157 FP=5 TN=395 FN=19
- **body_of_stomach_normal**: TP=191 FP=6 TN=378 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 5 | 3 |
| Malignant_lesions | 16 | 157 | 3 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9217    0.9615    0.9412       208
     Malignant_lesions     0.9691    0.8920    0.9290       176
body_of_stomach_normal     0.9695    0.9948    0.9820       192

              accuracy                         0.9514       576
             macro avg     0.9534    0.9495    0.9507       576
          weighted avg     0.9521    0.9514    0.9511       576

```