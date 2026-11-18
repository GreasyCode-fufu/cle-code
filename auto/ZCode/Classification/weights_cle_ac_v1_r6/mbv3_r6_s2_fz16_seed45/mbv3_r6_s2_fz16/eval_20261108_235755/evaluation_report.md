# mbv3_r6_s2_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_fz16_seed45/mbv3_r6_s2_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9647 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9977 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9721 |
| Balanced Acc | 0.9690 |
| Macro-Sensitivity | 0.9690 |
| Macro-Specificity | 0.9866 |
| Macro-Precision | 0.9609 |
| AUC-PR (Macro) | 0.9930 |
| Cohen's Kappa | 0.9555 |
| MCC | 0.9556 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 3041.3 (0.33 ms/张) |
| End-to-End FPS | 83.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9641 | 0.9828 | 0.9792 | 0.9716 | 0.9702 | 0.9976 | 0.9972 | 195 |
| Malignant_lesions | 0.9494 | 0.9770 | 0.9036 | 0.9259 | 0.9884 | 0.9956 | 0.9818 | 79 |
| body_of_stomach_normal | 0.9935 | 1.0000 | 1.0000 | 0.9967 | 0.9964 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=4 TN=228 FN=7
- **Malignant_lesions**: TP=75 FP=8 TN=340 FN=4
- **body_of_stomach_normal**: TP=152 FP=0 TN=274 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 7 | 0 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 1 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9792    0.9641    0.9716       195
     Malignant_lesions     0.9036    0.9494    0.9259        79
body_of_stomach_normal     1.0000    0.9935    0.9967       153

              accuracy                         0.9719       427
             macro avg     0.9609    0.9690    0.9647       427
          weighted avg     0.9727    0.9719    0.9721       427

```