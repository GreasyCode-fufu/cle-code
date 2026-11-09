# mbv3_sft_b2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b2_seed45/mbv3_sft_b2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9509 |
| Accuracy | 0.9602 |
| Macro-AUROC (OvR) | 0.9951 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9607 |
| Balanced Acc | 0.9584 |
| Macro-Sensitivity | 0.9584 |
| Macro-Specificity | 0.9813 |
| Macro-Precision | 0.9446 |
| AUC-PR (Macro) | 0.9858 |
| Cohen's Kappa | 0.9371 |
| MCC | 0.9376 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2646.4 (0.38 ms/张) |
| End-to-End FPS | 177.1 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9385 | 0.9784 | 0.9734 | 0.9556 | 0.9498 | 0.9943 | 0.9934 | 195 |
| Malignant_lesions | 0.9367 | 0.9655 | 0.8605 | 0.8970 | 0.9853 | 0.9909 | 0.9641 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=183 FP=5 TN=227 FN=12
- **Malignant_lesions**: TP=74 FP=12 TN=336 FN=5
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 183 | 12 | 0 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9734    0.9385    0.9556       195
     Malignant_lesions     0.8605    0.9367    0.8970        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9602       427
             macro avg     0.9446    0.9584    0.9509       427
          weighted avg     0.9620    0.9602    0.9607       427

```