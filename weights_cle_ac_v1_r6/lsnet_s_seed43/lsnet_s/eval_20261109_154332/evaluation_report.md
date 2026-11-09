# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/lsnet_s_seed43/lsnet_s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9207 |
| Accuracy | 0.9219 |
| Macro-AUROC (OvR) | 0.9876 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9213 |
| Balanced Acc | 0.9185 |
| Macro-Sensitivity | 0.9185 |
| Macro-Specificity | 0.9601 |
| Macro-Precision | 0.9271 |
| AUC-PR (Macro) | 0.9769 |
| Cohen's Kappa | 0.8822 |
| MCC | 0.8840 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 5631.4 (0.18 ms/张) |
| End-to-End FPS | 294.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9239 | 0.8767 | 0.9149 | 0.9742 | 0.9842 | 0.9743 | 208 |
| Malignant_lesions | 0.8352 | 0.9850 | 0.9608 | 0.8936 | 0.9314 | 0.9814 | 0.9621 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9714 | 0.9439 | 0.9536 | 0.9816 | 0.9971 | 0.9943 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=28 TN=340 FN=9
- **Malignant_lesions**: TP=147 FP=6 TN=394 FN=29
- **body_of_stomach_normal**: TP=185 FP=11 TN=373 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 5 | 4 |
| Malignant_lesions | 22 | 147 | 7 |
| body_of_stomach_normal | 6 | 1 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8767    0.9567    0.9149       208
     Malignant_lesions     0.9608    0.8352    0.8936       176
body_of_stomach_normal     0.9439    0.9635    0.9536       192

              accuracy                         0.9219       576
             macro avg     0.9271    0.9185    0.9207       576
          weighted avg     0.9248    0.9219    0.9213       576

```