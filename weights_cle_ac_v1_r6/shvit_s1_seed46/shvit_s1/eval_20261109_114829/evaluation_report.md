# shvit_s1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/shvit_s1_seed46/shvit_s1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9422 |
| Accuracy | 0.9532 |
| Macro-AUROC (OvR) | 0.9945 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9530 |
| Balanced Acc | 0.9432 |
| Macro-Sensitivity | 0.9432 |
| Macro-Specificity | 0.9765 |
| Macro-Precision | 0.9413 |
| AUC-PR (Macro) | 0.9844 |
| Cohen's Kappa | 0.9256 |
| MCC | 0.9258 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 245.71M |
| FLOPs | 491.41M |
| Params | 6.01M |
| Pure-Inference FPS | 2985.2 (0.33 ms/张) |
| End-to-End FPS | 204.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9436 | 0.9698 | 0.9634 | 0.9534 | 0.9534 | 0.9934 | 0.9924 | 195 |
| Malignant_lesions | 0.8861 | 0.9741 | 0.8861 | 0.8861 | 0.9741 | 0.9904 | 0.9611 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9854 | 0.9745 | 0.9871 | 1.0000 | 0.9998 | 0.9997 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=184 FP=7 TN=225 FN=11
- **Malignant_lesions**: TP=70 FP=9 TN=339 FN=9
- **body_of_stomach_normal**: TP=153 FP=4 TN=270 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 184 | 9 | 2 |
| Malignant_lesions | 7 | 70 | 2 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9634    0.9436    0.9534       195
     Malignant_lesions     0.8861    0.8861    0.8861        79
body_of_stomach_normal     0.9745    1.0000    0.9871       153

              accuracy                         0.9532       427
             macro avg     0.9413    0.9432    0.9422       427
          weighted avg     0.9531    0.9532    0.9530       427

```