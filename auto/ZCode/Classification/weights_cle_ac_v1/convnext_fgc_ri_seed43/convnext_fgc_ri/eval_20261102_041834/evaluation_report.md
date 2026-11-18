# convnext_fgc_ri 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_ri_seed43/convnext_fgc_ri/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9495 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9496 |
| Balanced Acc | 0.9515 |
| Macro-Sensitivity | 0.9515 |
| Macro-Specificity | 0.9753 |
| Macro-Precision | 0.9493 |
| AUC-PR (Macro) | 0.9925 |
| Cohen's Kappa | 0.9245 |
| MCC | 0.9254 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 1044.6 (0.96 ms/张) |
| End-to-End FPS | 237.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8990 | 0.9864 | 0.9740 | 0.9350 | 0.9453 | 0.9947 | 0.9911 | 208 |
| Malignant_lesions | 0.9659 | 0.9525 | 0.8995 | 0.9315 | 0.9845 | 0.9941 | 0.9872 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=5 TN=363 FN=21
- **Malignant_lesions**: TP=170 FP=19 TN=381 FN=6
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 18 | 3 |
| Malignant_lesions | 4 | 170 | 2 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9740    0.8990    0.9350       208
     Malignant_lesions     0.8995    0.9659    0.9315       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9497       576
             macro avg     0.9493    0.9515    0.9495       576
          weighted avg     0.9513    0.9497    0.9496       576

```