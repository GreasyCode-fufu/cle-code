# mbv3_sft_b1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b1_seed46/mbv3_sft_b1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9566 |
| Accuracy | 0.9649 |
| Macro-AUROC (OvR) | 0.9945 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9650 |
| Balanced Acc | 0.9588 |
| Macro-Sensitivity | 0.9588 |
| Macro-Specificity | 0.9825 |
| Macro-Precision | 0.9545 |
| AUC-PR (Macro) | 0.9824 |
| Cohen's Kappa | 0.9442 |
| MCC | 0.9443 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2789.5 (0.36 ms/张) |
| End-to-End FPS | 103.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9741 | 0.9689 | 0.9639 | 0.9658 | 0.9947 | 0.9942 | 195 |
| Malignant_lesions | 0.9241 | 0.9770 | 0.9012 | 0.9125 | 0.9827 | 0.9892 | 0.9537 | 79 |
| body_of_stomach_normal | 0.9935 | 0.9964 | 0.9935 | 0.9935 | 0.9964 | 0.9997 | 0.9995 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=6 TN=226 FN=8
- **Malignant_lesions**: TP=73 FP=8 TN=340 FN=6
- **body_of_stomach_normal**: TP=152 FP=1 TN=273 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 8 | 0 |
| Malignant_lesions | 5 | 73 | 1 |
| body_of_stomach_normal | 1 | 0 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9689    0.9590    0.9639       195
     Malignant_lesions     0.9012    0.9241    0.9125        79
body_of_stomach_normal     0.9935    0.9935    0.9935       153

              accuracy                         0.9649       427
             macro avg     0.9545    0.9588    0.9566       427
          weighted avg     0.9652    0.9649    0.9650       427

```