# mbv3_sft_b1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b1_seed43/mbv3_sft_b1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9536 |
| Accuracy | 0.9602 |
| Macro-AUROC (OvR) | 0.9957 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9607 |
| Balanced Acc | 0.9620 |
| Macro-Sensitivity | 0.9620 |
| Macro-Specificity | 0.9811 |
| Macro-Precision | 0.9468 |
| AUC-PR (Macro) | 0.9876 |
| Cohen's Kappa | 0.9372 |
| MCC | 0.9376 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1790.7 (0.56 ms/张) |
| End-to-End FPS | 100.7 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9436 | 0.9784 | 0.9735 | 0.9583 | 0.9538 | 0.9943 | 0.9938 | 195 |
| Malignant_lesions | 0.9620 | 0.9684 | 0.8736 | 0.9157 | 0.9912 | 0.9931 | 0.9692 | 79 |
| body_of_stomach_normal | 0.9804 | 0.9964 | 0.9934 | 0.9868 | 0.9891 | 0.9999 | 0.9997 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=184 FP=5 TN=227 FN=11
- **Malignant_lesions**: TP=76 FP=11 TN=337 FN=3
- **body_of_stomach_normal**: TP=150 FP=1 TN=273 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 184 | 11 | 0 |
| Malignant_lesions | 2 | 76 | 1 |
| body_of_stomach_normal | 3 | 0 | 150 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9735    0.9436    0.9583       195
     Malignant_lesions     0.8736    0.9620    0.9157        79
body_of_stomach_normal     0.9934    0.9804    0.9868       153

              accuracy                         0.9602       427
             macro avg     0.9468    0.9620    0.9536       427
          weighted avg     0.9622    0.9602    0.9607       427

```