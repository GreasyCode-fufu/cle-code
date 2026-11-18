# mbv3_sft_b4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b4_seed46/mbv3_sft_b4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9211 |
| Accuracy | 0.9219 |
| Macro-AUROC (OvR) | 0.9916 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9216 |
| Balanced Acc | 0.9178 |
| Macro-Sensitivity | 0.9178 |
| Macro-Specificity | 0.9598 |
| Macro-Precision | 0.9308 |
| AUC-PR (Macro) | 0.9841 |
| Cohen's Kappa | 0.8821 |
| MCC | 0.8852 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4208.6 (0.24 ms/张) |
| End-to-End FPS | 179.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9049 | 0.8523 | 0.9079 | 0.9823 | 0.9876 | 0.9769 | 208 |
| Malignant_lesions | 0.8239 | 0.9875 | 0.9667 | 0.8896 | 0.9272 | 0.9884 | 0.9780 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9870 | 0.9735 | 0.9659 | 0.9793 | 0.9987 | 0.9975 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=35 TN=333 FN=6
- **Malignant_lesions**: TP=145 FP=5 TN=395 FN=31
- **body_of_stomach_normal**: TP=184 FP=5 TN=379 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 28 | 145 | 3 |
| body_of_stomach_normal | 7 | 1 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8523    0.9712    0.9079       208
     Malignant_lesions     0.9667    0.8239    0.8896       176
body_of_stomach_normal     0.9735    0.9583    0.9659       192

              accuracy                         0.9219       576
             macro avg     0.9308    0.9178    0.9211       576
          weighted avg     0.9277    0.9219    0.9216       576

```