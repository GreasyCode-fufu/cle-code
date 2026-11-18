# convnext_tiny 评估报告

- 权重: `/home/login/data/cle/auto/Network/CLE-Net/historical_protocol_codex_20261109/runs/convnext_tiny_seed42/convnext_tiny/model_best.pth`
- 测试集: `/home/login/data/cle/auto/Network/CLE-Net/datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9791 |
| Accuracy | 0.9836 |
| Macro-AUROC (OvR) | 0.9988 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9837 |
| Balanced Acc | 0.9800 |
| Macro-Sensitivity | 0.9800 |
| Macro-Specificity | 0.9919 |
| Macro-Precision | 0.9782 |
| AUC-PR (Macro) | 0.9961 |
| Cohen's Kappa | 0.9739 |
| MCC | 0.9740 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1663.4 (0.60 ms/张) |
| End-to-End FPS | 264.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9871 | 0.9846 | 0.9846 | 0.9871 | 0.9988 | 0.9986 | 195 |
| Malignant_lesions | 0.9620 | 0.9885 | 0.9500 | 0.9560 | 0.9914 | 0.9976 | 0.9898 | 79 |
| body_of_stomach_normal | 0.9935 | 1.0000 | 1.0000 | 0.9967 | 0.9964 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=3 TN=229 FN=3
- **Malignant_lesions**: TP=76 FP=4 TN=344 FN=3
- **body_of_stomach_normal**: TP=152 FP=0 TN=274 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 3 | 0 |
| Malignant_lesions | 3 | 76 | 0 |
| body_of_stomach_normal | 0 | 1 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9846    0.9846    0.9846       195
     Malignant_lesions     0.9500    0.9620    0.9560        79
body_of_stomach_normal     1.0000    0.9935    0.9967       153

              accuracy                         0.9836       427
             macro avg     0.9782    0.9800    0.9791       427
          weighted avg     0.9837    0.9836    0.9837       427

```