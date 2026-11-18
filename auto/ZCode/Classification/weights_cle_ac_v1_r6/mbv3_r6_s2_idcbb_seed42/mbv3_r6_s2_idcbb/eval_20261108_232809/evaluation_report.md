# mbv3_r6_s2_idcbb 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_idcbb_seed42/mbv3_r6_s2_idcbb/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9317 |
| Accuracy | 0.9323 |
| Macro-AUROC (OvR) | 0.9895 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9321 |
| Balanced Acc | 0.9317 |
| Macro-Sensitivity | 0.9317 |
| Macro-Specificity | 0.9660 |
| Macro-Precision | 0.9318 |
| AUC-PR (Macro) | 0.9800 |
| Cohen's Kappa | 0.8982 |
| MCC | 0.8982 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 234.73M |
| FLOPs | 469.46M |
| Params | 3.26M |
| Pure-Inference FPS | 3054.6 (0.33 ms/张) |
| End-to-End FPS | 117.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9135 | 0.9511 | 0.9135 | 0.9135 | 0.9511 | 0.9853 | 0.9731 | 208 |
| Malignant_lesions | 0.8920 | 0.9600 | 0.9075 | 0.8997 | 0.9529 | 0.9839 | 0.9682 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9993 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=18 TN=350 FN=18
- **Malignant_lesions**: TP=157 FP=16 TN=384 FN=19
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 15 | 3 |
| Malignant_lesions | 17 | 157 | 2 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9135    0.9135    0.9135       208
     Malignant_lesions     0.9075    0.8920    0.8997       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9323       576
             macro avg     0.9318    0.9317    0.9317       576
          weighted avg     0.9319    0.9323    0.9321       576

```