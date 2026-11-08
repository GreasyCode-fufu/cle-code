# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz16bn_seed45/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9619 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9969 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9697 |
| Balanced Acc | 0.9652 |
| Macro-Sensitivity | 0.9652 |
| Macro-Specificity | 0.9852 |
| Macro-Precision | 0.9588 |
| AUC-PR (Macro) | 0.9909 |
| Cohen's Kappa | 0.9517 |
| MCC | 0.9518 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3408.5 (0.29 ms/张) |
| End-to-End FPS | 110.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9784 | 0.9740 | 0.9664 | 0.9660 | 0.9965 | 0.9960 | 195 |
| Malignant_lesions | 0.9367 | 0.9770 | 0.9024 | 0.9193 | 0.9855 | 0.9943 | 0.9767 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=5 TN=227 FN=8
- **Malignant_lesions**: TP=74 FP=8 TN=340 FN=5
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 8 | 0 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9740    0.9590    0.9664       195
     Malignant_lesions     0.9024    0.9367    0.9193        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9696       427
             macro avg     0.9588    0.9652    0.9619       427
          weighted avg     0.9701    0.9696    0.9697       427

```