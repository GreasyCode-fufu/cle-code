# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/vit_base_patch16_224_seed42/vit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8998 |
| Accuracy | 0.9010 |
| Macro-AUROC (OvR) | 0.9844 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9004 |
| Balanced Acc | 0.8964 |
| Macro-Sensitivity | 0.8964 |
| Macro-Specificity | 0.9491 |
| Macro-Precision | 0.9125 |
| AUC-PR (Macro) | 0.9716 |
| Cohen's Kappa | 0.8505 |
| MCC | 0.8547 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 171.5 (5.83 ms/张) |
| End-to-End FPS | 109.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.8832 | 0.8208 | 0.8795 | 0.9673 | 0.9740 | 0.9632 | 208 |
| Malignant_lesions | 0.7784 | 0.9850 | 0.9580 | 0.8589 | 0.9099 | 0.9846 | 0.9612 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9792 | 0.9585 | 0.9610 | 0.9817 | 0.9947 | 0.9904 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=43 TN=325 FN=11
- **Malignant_lesions**: TP=137 FP=6 TN=394 FN=39
- **body_of_stomach_normal**: TP=185 FP=8 TN=376 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 6 | 5 |
| Malignant_lesions | 36 | 137 | 3 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8208    0.9471    0.8795       208
     Malignant_lesions     0.9580    0.7784    0.8589       176
body_of_stomach_normal     0.9585    0.9635    0.9610       192

              accuracy                         0.9010       576
             macro avg     0.9125    0.8964    0.8998       576
          weighted avg     0.9087    0.9010    0.9004       576

```