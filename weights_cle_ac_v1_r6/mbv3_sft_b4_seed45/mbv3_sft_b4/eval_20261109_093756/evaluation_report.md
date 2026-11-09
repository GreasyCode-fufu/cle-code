# mbv3_sft_b4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b4_seed45/mbv3_sft_b4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9317 |
| Accuracy | 0.9323 |
| Macro-AUROC (OvR) | 0.9897 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9321 |
| Balanced Acc | 0.9293 |
| Macro-Sensitivity | 0.9293 |
| Macro-Specificity | 0.9653 |
| Macro-Precision | 0.9376 |
| AUC-PR (Macro) | 0.9804 |
| Cohen's Kappa | 0.8979 |
| MCC | 0.8996 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3664.5 (0.27 ms/张) |
| End-to-End FPS | 176.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9239 | 0.8772 | 0.9174 | 0.9770 | 0.9852 | 0.9762 | 208 |
| Malignant_lesions | 0.8523 | 0.9850 | 0.9615 | 0.9036 | 0.9381 | 0.9854 | 0.9681 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9870 | 0.9740 | 0.9740 | 0.9870 | 0.9984 | 0.9969 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=28 TN=340 FN=8
- **Malignant_lesions**: TP=150 FP=6 TN=394 FN=26
- **body_of_stomach_normal**: TP=187 FP=5 TN=379 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 23 | 150 | 3 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8772    0.9615    0.9174       208
     Malignant_lesions     0.9615    0.8523    0.9036       176
body_of_stomach_normal     0.9740    0.9740    0.9740       192

              accuracy                         0.9323       576
             macro avg     0.9376    0.9293    0.9317       576
          weighted avg     0.9352    0.9323    0.9321       576

```