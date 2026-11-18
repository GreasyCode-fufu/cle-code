# convnext_tiny 评估报告

- 权重: `/home/login/data/cle/auto/Network/CLE-Net/historical_protocol_codex_20261109/runs/convnext_tiny_seed43/convnext_tiny/model_best.pth`
- 测试集: `/home/login/data/cle/auto/Network/CLE-Net/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9597 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9962 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9600 |
| Balanced Acc | 0.9608 |
| Macro-Sensitivity | 0.9608 |
| Macro-Specificity | 0.9803 |
| Macro-Precision | 0.9592 |
| AUC-PR (Macro) | 0.9922 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9403 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1716.0 (0.58 ms/张) |
| End-to-End FPS | 267.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9864 | 0.9747 | 0.9507 | 0.9603 | 0.9949 | 0.9920 | 208 |
| Malignant_lesions | 0.9545 | 0.9675 | 0.9282 | 0.9412 | 0.9797 | 0.9939 | 0.9850 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9870 | 0.9746 | 0.9871 | 1.0000 | 0.9999 | 0.9997 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=5 TN=363 FN=15
- **Malignant_lesions**: TP=168 FP=13 TN=387 FN=8
- **body_of_stomach_normal**: TP=192 FP=5 TN=379 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 13 | 2 |
| Malignant_lesions | 5 | 168 | 3 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9747    0.9279    0.9507       208
     Malignant_lesions     0.9282    0.9545    0.9412       176
body_of_stomach_normal     0.9746    1.0000    0.9871       192

              accuracy                         0.9601       576
             macro avg     0.9592    0.9608    0.9597       576
          weighted avg     0.9605    0.9601    0.9600       576

```