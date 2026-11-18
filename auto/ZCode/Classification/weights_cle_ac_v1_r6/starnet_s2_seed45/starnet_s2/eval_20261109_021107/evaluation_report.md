# starnet_s2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_seed45/starnet_s2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9250 |
| Accuracy | 0.9253 |
| Macro-AUROC (OvR) | 0.9883 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9252 |
| Balanced Acc | 0.9231 |
| Macro-Sensitivity | 0.9231 |
| Macro-Specificity | 0.9619 |
| Macro-Precision | 0.9290 |
| AUC-PR (Macro) | 0.9775 |
| Cohen's Kappa | 0.8875 |
| MCC | 0.8885 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.29M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 2200.7 (0.45 ms/张) |
| End-to-End FPS | 224.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9239 | 0.8744 | 0.9049 | 0.9632 | 0.9822 | 0.9738 | 208 |
| Malignant_lesions | 0.8580 | 0.9775 | 0.9437 | 0.8988 | 0.9399 | 0.9839 | 0.9609 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9844 | 0.9689 | 0.9714 | 0.9869 | 0.9989 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=28 TN=340 FN=13
- **Malignant_lesions**: TP=151 FP=9 TN=391 FN=25
- **body_of_stomach_normal**: TP=187 FP=6 TN=378 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 9 | 4 |
| Malignant_lesions | 23 | 151 | 2 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8744    0.9375    0.9049       208
     Malignant_lesions     0.9437    0.8580    0.8988       176
body_of_stomach_normal     0.9689    0.9740    0.9714       192

              accuracy                         0.9253       576
             macro avg     0.9290    0.9231    0.9250       576
          weighted avg     0.9271    0.9253    0.9252       576

```