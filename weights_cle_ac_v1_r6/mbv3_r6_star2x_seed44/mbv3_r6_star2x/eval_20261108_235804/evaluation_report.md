# mbv3_r6_star2x 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star2x_seed44/mbv3_r6_star2x/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9422 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9938 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9426 |
| Balanced Acc | 0.9405 |
| Macro-Sensitivity | 0.9405 |
| Macro-Specificity | 0.9708 |
| Macro-Precision | 0.9454 |
| AUC-PR (Macro) | 0.9879 |
| Cohen's Kappa | 0.9137 |
| MCC | 0.9145 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.10M |
| FLOPs | 464.20M |
| Params | 3.39M |
| Pure-Inference FPS | 3912.1 (0.26 ms/张) |
| End-to-End FPS | 109.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9429 | 0.9054 | 0.9349 | 0.9802 | 0.9895 | 0.9817 | 208 |
| Malignant_lesions | 0.8864 | 0.9825 | 0.9571 | 0.9204 | 0.9516 | 0.9930 | 0.9843 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9870 | 0.9738 | 0.9713 | 0.9844 | 0.9988 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=21 TN=347 FN=7
- **Malignant_lesions**: TP=156 FP=7 TN=393 FN=20
- **body_of_stomach_normal**: TP=186 FP=5 TN=379 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 17 | 156 | 3 |
| body_of_stomach_normal | 4 | 2 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9054    0.9663    0.9349       208
     Malignant_lesions     0.9571    0.8864    0.9204       176
body_of_stomach_normal     0.9738    0.9688    0.9713       192

              accuracy                         0.9427       576
             macro avg     0.9454    0.9405    0.9422       576
          weighted avg     0.9440    0.9427    0.9426       576

```