# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce4s_seed46/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9565 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9941 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9566 |
| Balanced Acc | 0.9572 |
| Macro-Sensitivity | 0.9572 |
| Macro-Specificity | 0.9784 |
| Macro-Precision | 0.9561 |
| AUC-PR (Macro) | 0.9886 |
| Cohen's Kappa | 0.9348 |
| MCC | 0.9349 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5915.5 (0.17 ms/张) |
| End-to-End FPS | 242.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9783 | 0.9606 | 0.9489 | 0.9651 | 0.9920 | 0.9878 | 208 |
| Malignant_lesions | 0.9602 | 0.9675 | 0.9286 | 0.9441 | 0.9822 | 0.9942 | 0.9836 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9896 | 0.9791 | 0.9765 | 0.9870 | 0.9961 | 0.9944 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=8 TN=360 FN=13
- **Malignant_lesions**: TP=169 FP=13 TN=387 FN=7
- **body_of_stomach_normal**: TP=187 FP=4 TN=380 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 12 | 1 |
| Malignant_lesions | 4 | 169 | 3 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9606    0.9375    0.9489       208
     Malignant_lesions     0.9286    0.9602    0.9441       176
body_of_stomach_normal     0.9791    0.9740    0.9765       192

              accuracy                         0.9566       576
             macro avg     0.9561    0.9572    0.9565       576
          weighted avg     0.9570    0.9566    0.9566       576

```