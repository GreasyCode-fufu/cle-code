# convnext_tiny 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/cle_net_rerun_codex_20261109_v2/results/convnext_tiny_seed43/convnext_tiny/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9532 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9533 |
| Balanced Acc | 0.9537 |
| Macro-Sensitivity | 0.9537 |
| Macro-Specificity | 0.9767 |
| Macro-Precision | 0.9537 |
| AUC-PR (Macro) | 0.9904 |
| Cohen's Kappa | 0.9296 |
| MCC | 0.9300 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1720.0 (0.58 ms/张) |
| End-to-End FPS | 248.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9701 | 0.9474 | 0.9496 | 0.9728 | 0.9931 | 0.9901 | 208 |
| Malignant_lesions | 0.9716 | 0.9625 | 0.9194 | 0.9448 | 0.9872 | 0.9932 | 0.9839 | 176 |
| body_of_stomach_normal | 0.9375 | 0.9974 | 0.9945 | 0.9651 | 0.9696 | 0.9985 | 0.9971 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=11 TN=357 FN=10
- **Malignant_lesions**: TP=171 FP=15 TN=385 FN=5
- **body_of_stomach_normal**: TP=180 FP=1 TN=383 FN=12

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 10 | 0 |
| Malignant_lesions | 4 | 171 | 1 |
| body_of_stomach_normal | 7 | 5 | 180 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9474    0.9519    0.9496       208
     Malignant_lesions     0.9194    0.9716    0.9448       176
body_of_stomach_normal     0.9945    0.9375    0.9651       192

              accuracy                         0.9531       576
             macro avg     0.9537    0.9537    0.9532       576
          weighted avg     0.9545    0.9531    0.9533       576

```