# mbv3_sft_head 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_head_seed42/mbv3_sft_head/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9211 |
| Accuracy | 0.9344 |
| Macro-AUROC (OvR) | 0.9906 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9351 |
| Balanced Acc | 0.9316 |
| Macro-Sensitivity | 0.9316 |
| Macro-Specificity | 0.9695 |
| Macro-Precision | 0.9135 |
| AUC-PR (Macro) | 0.9651 |
| Cohen's Kappa | 0.8969 |
| MCC | 0.8983 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2022.1 (0.49 ms/张) |
| End-to-End FPS | 27.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9026 | 0.9828 | 0.9778 | 0.9387 | 0.9231 | 0.9897 | 0.9897 | 195 |
| Malignant_lesions | 0.8987 | 0.9511 | 0.8068 | 0.8503 | 0.9764 | 0.9822 | 0.9060 | 79 |
| body_of_stomach_normal | 0.9935 | 0.9745 | 0.9560 | 0.9744 | 0.9963 | 0.9998 | 0.9996 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=176 FP=4 TN=228 FN=19
- **Malignant_lesions**: TP=71 FP=17 TN=331 FN=8
- **body_of_stomach_normal**: TP=152 FP=7 TN=267 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 176 | 16 | 3 |
| Malignant_lesions | 4 | 71 | 4 |
| body_of_stomach_normal | 0 | 1 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9778    0.9026    0.9387       195
     Malignant_lesions     0.8068    0.8987    0.8503        79
body_of_stomach_normal     0.9560    0.9935    0.9744       153

              accuracy                         0.9344       427
             macro avg     0.9135    0.9316    0.9211       427
          weighted avg     0.9383    0.9344    0.9351       427

```