# deit_tiny_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/deit_tiny_patch16_224_seed44/deit_tiny_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9324 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9903 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9334 |
| Balanced Acc | 0.9318 |
| Macro-Sensitivity | 0.9318 |
| Macro-Specificity | 0.9668 |
| Macro-Precision | 0.9352 |
| AUC-PR (Macro) | 0.9815 |
| Cohen's Kappa | 0.9007 |
| MCC | 0.9016 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 5.49M |
| Pure-Inference FPS | 3019.7 (0.33 ms/张) |
| End-to-End FPS | 248.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9620 | 0.9333 | 0.9378 | 0.9672 | 0.9858 | 0.9810 | 208 |
| Malignant_lesions | 0.8636 | 0.9800 | 0.9500 | 0.9048 | 0.9423 | 0.9871 | 0.9675 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9583 | 0.9223 | 0.9548 | 0.9946 | 0.9980 | 0.9960 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=14 TN=354 FN=12
- **Malignant_lesions**: TP=152 FP=8 TN=392 FN=24
- **body_of_stomach_normal**: TP=190 FP=16 TN=368 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 8 | 4 |
| Malignant_lesions | 12 | 152 | 12 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9333    0.9423    0.9378       208
     Malignant_lesions     0.9500    0.8636    0.9048       176
body_of_stomach_normal     0.9223    0.9896    0.9548       192

              accuracy                         0.9340       576
             macro avg     0.9352    0.9318    0.9324       576
          weighted avg     0.9348    0.9340    0.9334       576

```