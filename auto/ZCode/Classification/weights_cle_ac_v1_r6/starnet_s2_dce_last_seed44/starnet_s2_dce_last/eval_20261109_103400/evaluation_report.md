# starnet_s2_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce_last_seed44/starnet_s2_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9371 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9933 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9373 |
| Balanced Acc | 0.9357 |
| Macro-Sensitivity | 0.9357 |
| Macro-Specificity | 0.9682 |
| Macro-Precision | 0.9397 |
| AUC-PR (Macro) | 0.9873 |
| Cohen's Kappa | 0.9059 |
| MCC | 0.9064 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 5324.5 (0.19 ms/张) |
| End-to-End FPS | 8.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9457 | 0.9083 | 0.9296 | 0.9721 | 0.9903 | 0.9829 | 208 |
| Malignant_lesions | 0.8864 | 0.9825 | 0.9571 | 0.9204 | 0.9516 | 0.9913 | 0.9826 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9766 | 0.9538 | 0.9612 | 0.9843 | 0.9981 | 0.9963 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=20 TN=348 FN=10
- **Malignant_lesions**: TP=156 FP=7 TN=393 FN=20
- **body_of_stomach_normal**: TP=186 FP=9 TN=375 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 14 | 156 | 6 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9083    0.9519    0.9296       208
     Malignant_lesions     0.9571    0.8864    0.9204       176
body_of_stomach_normal     0.9538    0.9688    0.9612       192

              accuracy                         0.9375       576
             macro avg     0.9397    0.9357    0.9371       576
          weighted avg     0.9384    0.9375    0.9373       576

```