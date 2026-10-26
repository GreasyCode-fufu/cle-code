# deit_tiny_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/deit_tiny_patch16_224_seed42/deit_tiny_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9512 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9930 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9513 |
| Balanced Acc | 0.9493 |
| Macro-Sensitivity | 0.9493 |
| Macro-Specificity | 0.9750 |
| Macro-Precision | 0.9550 |
| AUC-PR (Macro) | 0.9871 |
| Cohen's Kappa | 0.9268 |
| MCC | 0.9277 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 5.49M |
| Pure-Inference FPS | 3540.6 (0.28 ms/张) |
| End-to-End FPS | 254.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9429 | 0.9054 | 0.9349 | 0.9802 | 0.9887 | 0.9829 | 208 |
| Malignant_lesions | 0.8920 | 0.9900 | 0.9752 | 0.9318 | 0.9542 | 0.9912 | 0.9803 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9922 | 0.9845 | 0.9870 | 0.9948 | 0.9991 | 0.9980 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=21 TN=347 FN=7
- **Malignant_lesions**: TP=157 FP=4 TN=396 FN=19
- **body_of_stomach_normal**: TP=190 FP=3 TN=381 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 4 | 3 |
| Malignant_lesions | 19 | 157 | 0 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9054    0.9663    0.9349       208
     Malignant_lesions     0.9752    0.8920    0.9318       176
body_of_stomach_normal     0.9845    0.9896    0.9870       192

              accuracy                         0.9514       576
             macro avg     0.9550    0.9493    0.9512       576
          weighted avg     0.9531    0.9514    0.9513       576

```