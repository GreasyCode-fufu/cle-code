# convnext_egb_scg 评估报告

- 权重: `/home/login/data/cle/auto/Network/CLE-Net/historical_protocol_codex_20261109/runs/convnext_egb_scg_seed44/convnext_egb_scg/model_best.pth`
- 测试集: `/home/login/data/cle/auto/Network/CLE-Net/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9688 |
| Accuracy | 0.9688 |
| Macro-AUROC (OvR) | 0.9954 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9688 |
| Balanced Acc | 0.9683 |
| Macro-Sensitivity | 0.9683 |
| Macro-Specificity | 0.9842 |
| Macro-Precision | 0.9695 |
| AUC-PR (Macro) | 0.9909 |
| Cohen's Kappa | 0.9530 |
| MCC | 0.9531 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 1003.7 (1.00 ms/张) |
| End-to-End FPS | 242.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9701 | 0.9484 | 0.9596 | 0.9835 | 0.9940 | 0.9837 | 208 |
| Malignant_lesions | 0.9545 | 0.9850 | 0.9655 | 0.9600 | 0.9801 | 0.9926 | 0.9893 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9974 | 0.9947 | 0.9869 | 0.9897 | 0.9998 | 0.9996 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=11 TN=357 FN=6
- **Malignant_lesions**: TP=168 FP=6 TN=394 FN=8
- **body_of_stomach_normal**: TP=188 FP=1 TN=383 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 6 | 0 |
| Malignant_lesions | 7 | 168 | 1 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9484    0.9712    0.9596       208
     Malignant_lesions     0.9655    0.9545    0.9600       176
body_of_stomach_normal     0.9947    0.9792    0.9869       192

              accuracy                         0.9688       576
             macro avg     0.9695    0.9683    0.9688       576
          weighted avg     0.9691    0.9688    0.9688       576

```