# mbv3_sft_head 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_head_seed44/mbv3_sft_head/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9284 |
| Accuracy | 0.9391 |
| Macro-AUROC (OvR) | 0.9894 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9399 |
| Balanced Acc | 0.9361 |
| Macro-Sensitivity | 0.9361 |
| Macro-Specificity | 0.9703 |
| Macro-Precision | 0.9222 |
| AUC-PR (Macro) | 0.9671 |
| Cohen's Kappa | 0.9039 |
| MCC | 0.9043 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2017.1 (0.50 ms/张) |
| End-to-End FPS | 125.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9612 | 0.9524 | 0.9375 | 0.9370 | 0.9864 | 0.9857 | 195 |
| Malignant_lesions | 0.9114 | 0.9569 | 0.8276 | 0.8675 | 0.9794 | 0.9822 | 0.9166 | 79 |
| body_of_stomach_normal | 0.9739 | 0.9927 | 0.9868 | 0.9803 | 0.9855 | 0.9995 | 0.9991 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=180 FP=9 TN=223 FN=15
- **Malignant_lesions**: TP=72 FP=15 TN=333 FN=7
- **body_of_stomach_normal**: TP=149 FP=2 TN=272 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 180 | 14 | 1 |
| Malignant_lesions | 6 | 72 | 1 |
| body_of_stomach_normal | 3 | 1 | 149 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9524    0.9231    0.9375       195
     Malignant_lesions     0.8276    0.9114    0.8675        79
body_of_stomach_normal     0.9868    0.9739    0.9803       153

              accuracy                         0.9391       427
             macro avg     0.9222    0.9361    0.9284       427
          weighted avg     0.9416    0.9391    0.9399       427

```