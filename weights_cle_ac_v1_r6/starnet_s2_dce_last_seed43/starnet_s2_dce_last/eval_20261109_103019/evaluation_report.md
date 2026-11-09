# starnet_s2_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce_last_seed43/starnet_s2_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9655 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9719 |
| Balanced Acc | 0.9669 |
| Macro-Sensitivity | 0.9669 |
| Macro-Specificity | 0.9859 |
| Macro-Precision | 0.9642 |
| AUC-PR (Macro) | 0.9866 |
| Cohen's Kappa | 0.9554 |
| MCC | 0.9554 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 2647.1 (0.38 ms/张) |
| End-to-End FPS | 7.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9641 | 0.9784 | 0.9741 | 0.9691 | 0.9701 | 0.9952 | 0.9950 | 195 |
| Malignant_lesions | 0.9367 | 0.9828 | 0.9250 | 0.9308 | 0.9856 | 0.9932 | 0.9648 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=5 TN=227 FN=7
- **Malignant_lesions**: TP=74 FP=6 TN=342 FN=5
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 6 | 1 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9741    0.9641    0.9691       195
     Malignant_lesions     0.9250    0.9367    0.9308        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9719       427
             macro avg     0.9642    0.9669    0.9655       427
          weighted avg     0.9720    0.9719    0.9719       427

```