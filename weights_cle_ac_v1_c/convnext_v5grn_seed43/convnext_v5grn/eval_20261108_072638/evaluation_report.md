# convnext_v5grn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5grn_seed43/convnext_v5grn/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9686 |
| Accuracy | 0.9688 |
| Macro-AUROC (OvR) | 0.9965 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9687 |
| Balanced Acc | 0.9696 |
| Macro-Sensitivity | 0.9696 |
| Macro-Specificity | 0.9846 |
| Macro-Precision | 0.9680 |
| AUC-PR (Macro) | 0.9930 |
| Cohen's Kappa | 0.9531 |
| MCC | 0.9533 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 362.7 (2.76 ms/张) |
| End-to-End FPS | 164.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9891 | 0.9800 | 0.9608 | 0.9681 | 0.9952 | 0.9919 | 208 |
| Malignant_lesions | 0.9716 | 0.9725 | 0.9396 | 0.9553 | 0.9873 | 0.9949 | 0.9887 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9922 | 0.9845 | 0.9896 | 0.9974 | 0.9992 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=4 TN=364 FN=12
- **Malignant_lesions**: TP=171 FP=11 TN=389 FN=5
- **body_of_stomach_normal**: TP=191 FP=3 TN=381 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 11 | 1 |
| Malignant_lesions | 3 | 171 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9800    0.9423    0.9608       208
     Malignant_lesions     0.9396    0.9716    0.9553       176
body_of_stomach_normal     0.9845    0.9948    0.9896       192

              accuracy                         0.9688       576
             macro avg     0.9680    0.9696    0.9686       576
          weighted avg     0.9692    0.9688    0.9687       576

```