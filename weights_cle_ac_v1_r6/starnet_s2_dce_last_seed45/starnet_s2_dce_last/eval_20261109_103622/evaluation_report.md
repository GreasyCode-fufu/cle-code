# starnet_s2_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce_last_seed45/starnet_s2_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9336 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9892 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9338 |
| Balanced Acc | 0.9326 |
| Macro-Sensitivity | 0.9326 |
| Macro-Specificity | 0.9665 |
| Macro-Precision | 0.9356 |
| AUC-PR (Macro) | 0.9796 |
| Cohen's Kappa | 0.9007 |
| MCC | 0.9011 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 4992.4 (0.20 ms/张) |
| End-to-End FPS | 8.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9402 | 0.8981 | 0.9151 | 0.9611 | 0.9825 | 0.9739 | 208 |
| Malignant_lesions | 0.8807 | 0.9750 | 0.9394 | 0.9091 | 0.9489 | 0.9860 | 0.9667 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9844 | 0.9692 | 0.9767 | 0.9921 | 0.9991 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=22 TN=346 FN=14
- **Malignant_lesions**: TP=155 FP=10 TN=390 FN=21
- **body_of_stomach_normal**: TP=189 FP=6 TN=378 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 10 | 4 |
| Malignant_lesions | 19 | 155 | 2 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8981    0.9327    0.9151       208
     Malignant_lesions     0.9394    0.8807    0.9091       176
body_of_stomach_normal     0.9692    0.9844    0.9767       192

              accuracy                         0.9340       576
             macro avg     0.9356    0.9326    0.9336       576
          weighted avg     0.9344    0.9340    0.9338       576

```