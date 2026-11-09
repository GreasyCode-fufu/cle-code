# mbv3_sft_b2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b2_seed45/mbv3_sft_b2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9215 |
| Accuracy | 0.9236 |
| Macro-AUROC (OvR) | 0.9900 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9225 |
| Balanced Acc | 0.9183 |
| Macro-Sensitivity | 0.9183 |
| Macro-Specificity | 0.9606 |
| Macro-Precision | 0.9347 |
| AUC-PR (Macro) | 0.9814 |
| Cohen's Kappa | 0.8846 |
| MCC | 0.8891 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3300.1 (0.30 ms/张) |
| End-to-End FPS | 175.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9856 | 0.9049 | 0.8542 | 0.9152 | 0.9911 | 0.9865 | 0.9751 | 208 |
| Malignant_lesions | 0.7955 | 0.9950 | 0.9859 | 0.8805 | 0.9171 | 0.9846 | 0.9716 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9818 | 0.9639 | 0.9689 | 0.9869 | 0.9988 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=205 FP=35 TN=333 FN=3
- **Malignant_lesions**: TP=140 FP=2 TN=398 FN=36
- **body_of_stomach_normal**: TP=187 FP=7 TN=377 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 205 | 2 | 1 |
| Malignant_lesions | 30 | 140 | 6 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8542    0.9856    0.9152       208
     Malignant_lesions     0.9859    0.7955    0.8805       176
body_of_stomach_normal     0.9639    0.9740    0.9689       192

              accuracy                         0.9236       576
             macro avg     0.9347    0.9183    0.9215       576
          weighted avg     0.9310    0.9236    0.9225       576

```