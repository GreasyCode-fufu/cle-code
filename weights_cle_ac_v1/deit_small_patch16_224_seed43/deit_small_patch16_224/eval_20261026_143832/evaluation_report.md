# deit_small_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/deit_small_patch16_224_seed43/deit_small_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9483 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9948 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9482 |
| Balanced Acc | 0.9473 |
| Macro-Sensitivity | 0.9473 |
| Macro-Specificity | 0.9736 |
| Macro-Precision | 0.9504 |
| AUC-PR (Macro) | 0.9901 |
| Cohen's Kappa | 0.9216 |
| MCC | 0.9222 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.25G |
| FLOPs | 8.50G |
| Params | 21.59M |
| Pure-Inference FPS | 1097.1 (0.91 ms/张) |
| End-to-End FPS | 229.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9484 | 0.9136 | 0.9393 | 0.9803 | 0.9919 | 0.9865 | 208 |
| Malignant_lesions | 0.9432 | 0.9750 | 0.9432 | 0.9432 | 0.9750 | 0.9935 | 0.9855 | 176 |
| body_of_stomach_normal | 0.9323 | 0.9974 | 0.9944 | 0.9624 | 0.9672 | 0.9991 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=19 TN=349 FN=7
- **Malignant_lesions**: TP=166 FP=10 TN=390 FN=10
- **body_of_stomach_normal**: TP=179 FP=1 TN=383 FN=13

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 10 | 166 | 0 |
| body_of_stomach_normal | 9 | 4 | 179 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9136    0.9663    0.9393       208
     Malignant_lesions     0.9432    0.9432    0.9432       176
body_of_stomach_normal     0.9944    0.9323    0.9624       192

              accuracy                         0.9479       576
             macro avg     0.9504    0.9473    0.9483       576
          weighted avg     0.9496    0.9479    0.9482       576

```