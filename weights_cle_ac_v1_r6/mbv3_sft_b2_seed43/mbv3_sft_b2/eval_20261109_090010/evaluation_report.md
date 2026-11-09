# mbv3_sft_b2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b2_seed43/mbv3_sft_b2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8998 |
| Accuracy | 0.9028 |
| Macro-AUROC (OvR) | 0.9883 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9008 |
| Balanced Acc | 0.8976 |
| Macro-Sensitivity | 0.8976 |
| Macro-Specificity | 0.9501 |
| Macro-Precision | 0.9151 |
| AUC-PR (Macro) | 0.9783 |
| Cohen's Kappa | 0.8531 |
| MCC | 0.8581 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2985.4 (0.33 ms/张) |
| End-to-End FPS | 113.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9076 | 0.8528 | 0.8975 | 0.9681 | 0.9826 | 0.9688 | 208 |
| Malignant_lesions | 0.7614 | 0.9975 | 0.9926 | 0.8617 | 0.9048 | 0.9860 | 0.9728 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9453 | 0.9000 | 0.9403 | 0.9918 | 0.9964 | 0.9934 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=34 TN=334 FN=11
- **Malignant_lesions**: TP=134 FP=1 TN=399 FN=42
- **body_of_stomach_normal**: TP=189 FP=21 TN=363 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 1 | 10 |
| Malignant_lesions | 31 | 134 | 11 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8528    0.9471    0.8975       208
     Malignant_lesions     0.9926    0.7614    0.8617       176
body_of_stomach_normal     0.9000    0.9844    0.9403       192

              accuracy                         0.9028       576
             macro avg     0.9151    0.8976    0.8998       576
          weighted avg     0.9113    0.9028    0.9008       576

```