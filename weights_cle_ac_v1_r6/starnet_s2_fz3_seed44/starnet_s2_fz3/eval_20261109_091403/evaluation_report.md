# starnet_s2_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_seed44/starnet_s2_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9247 |
| Accuracy | 0.9253 |
| Macro-AUROC (OvR) | 0.9905 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9249 |
| Balanced Acc | 0.9239 |
| Macro-Sensitivity | 0.9239 |
| Macro-Specificity | 0.9622 |
| Macro-Precision | 0.9271 |
| AUC-PR (Macro) | 0.9820 |
| Cohen's Kappa | 0.8876 |
| MCC | 0.8882 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.29M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3630.5 (0.28 ms/张) |
| End-to-End FPS | 95.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9429 | 0.9014 | 0.9121 | 0.9559 | 0.9859 | 0.9764 | 208 |
| Malignant_lesions | 0.8693 | 0.9775 | 0.9444 | 0.9053 | 0.9444 | 0.9874 | 0.9728 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9661 | 0.9353 | 0.9567 | 0.9893 | 0.9983 | 0.9967 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=21 TN=347 FN=16
- **Malignant_lesions**: TP=153 FP=9 TN=391 FN=23
- **body_of_stomach_normal**: TP=188 FP=13 TN=371 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 9 | 7 |
| Malignant_lesions | 17 | 153 | 6 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9014    0.9231    0.9121       208
     Malignant_lesions     0.9444    0.8693    0.9053       176
body_of_stomach_normal     0.9353    0.9792    0.9567       192

              accuracy                         0.9253       576
             macro avg     0.9271    0.9239    0.9247       576
          weighted avg     0.9259    0.9253    0.9249       576

```