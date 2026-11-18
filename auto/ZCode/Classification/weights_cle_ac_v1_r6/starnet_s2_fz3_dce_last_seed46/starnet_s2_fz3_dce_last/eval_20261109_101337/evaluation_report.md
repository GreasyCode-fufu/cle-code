# starnet_s2_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce_last_seed46/starnet_s2_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9381 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9900 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9388 |
| Balanced Acc | 0.9372 |
| Macro-Sensitivity | 0.9372 |
| Macro-Specificity | 0.9693 |
| Macro-Precision | 0.9399 |
| AUC-PR (Macro) | 0.9809 |
| Cohen's Kappa | 0.9085 |
| MCC | 0.9090 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 4788.0 (0.21 ms/张) |
| End-to-End FPS | 7.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9511 | 0.9163 | 0.9314 | 0.9695 | 0.9848 | 0.9778 | 208 |
| Malignant_lesions | 0.8750 | 0.9750 | 0.9390 | 0.9059 | 0.9466 | 0.9864 | 0.9670 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9818 | 0.9645 | 0.9769 | 0.9947 | 0.9988 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=18 TN=350 FN=11
- **Malignant_lesions**: TP=154 FP=10 TN=390 FN=22
- **body_of_stomach_normal**: TP=190 FP=7 TN=377 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 17 | 154 | 5 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9163    0.9471    0.9314       208
     Malignant_lesions     0.9390    0.8750    0.9059       176
body_of_stomach_normal     0.9645    0.9896    0.9769       192

              accuracy                         0.9392       576
             macro avg     0.9399    0.9372    0.9381       576
          weighted avg     0.9393    0.9392    0.9388       576

```