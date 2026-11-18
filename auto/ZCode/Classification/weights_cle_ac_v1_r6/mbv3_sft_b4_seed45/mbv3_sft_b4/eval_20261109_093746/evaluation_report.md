# mbv3_sft_b4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b4_seed45/mbv3_sft_b4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9591 |
| Accuracy | 0.9672 |
| Macro-AUROC (OvR) | 0.9960 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9674 |
| Balanced Acc | 0.9635 |
| Macro-Sensitivity | 0.9635 |
| Macro-Specificity | 0.9842 |
| Macro-Precision | 0.9551 |
| AUC-PR (Macro) | 0.9878 |
| Cohen's Kappa | 0.9481 |
| MCC | 0.9482 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2519.8 (0.40 ms/张) |
| End-to-End FPS | 112.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9538 | 0.9784 | 0.9738 | 0.9637 | 0.9619 | 0.9954 | 0.9948 | 195 |
| Malignant_lesions | 0.9367 | 0.9741 | 0.8916 | 0.9136 | 0.9855 | 0.9924 | 0.9686 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=186 FP=5 TN=227 FN=9
- **Malignant_lesions**: TP=74 FP=9 TN=339 FN=5
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 186 | 9 | 0 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9738    0.9538    0.9637       195
     Malignant_lesions     0.8916    0.9367    0.9136        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9672       427
             macro avg     0.9551    0.9635    0.9591       427
          weighted avg     0.9680    0.9672    0.9674       427

```