# convnext_egb_scg 评估报告

- 权重: `/home/login/data/cle/auto/Network/CLE-Net/historical_protocol_codex_20261109/runs/convnext_egb_scg_seed42/convnext_egb_scg/model_best.pth`
- 测试集: `/home/login/data/cle/auto/Network/CLE-Net/datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9701 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9982 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9765 |
| Balanced Acc | 0.9678 |
| Macro-Sensitivity | 0.9678 |
| Macro-Specificity | 0.9875 |
| Macro-Precision | 0.9725 |
| AUC-PR (Macro) | 0.9940 |
| Cohen's Kappa | 0.9627 |
| MCC | 0.9627 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 986.7 (1.01 ms/张) |
| End-to-End FPS | 242.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9741 | 0.9695 | 0.9745 | 0.9826 | 0.9981 | 0.9978 | 195 |
| Malignant_lesions | 0.9241 | 0.9885 | 0.9481 | 0.9359 | 0.9829 | 0.9964 | 0.9843 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=6 TN=226 FN=4
- **Malignant_lesions**: TP=73 FP=4 TN=344 FN=6
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 4 | 0 |
| Malignant_lesions | 6 | 73 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9695    0.9795    0.9745       195
     Malignant_lesions     0.9481    0.9241    0.9359        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9766       427
             macro avg     0.9725    0.9678    0.9701       427
          weighted avg     0.9765    0.9766    0.9765       427

```