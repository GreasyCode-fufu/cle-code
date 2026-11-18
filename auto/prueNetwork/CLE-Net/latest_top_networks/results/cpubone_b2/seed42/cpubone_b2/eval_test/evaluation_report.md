# cpubone_b2 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/cpubone_b2/seed42/cpubone_b2/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9338 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9924 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9340 |
| Balanced Acc | 0.9326 |
| Macro-Sensitivity | 0.9326 |
| Macro-Specificity | 0.9666 |
| Macro-Precision | 0.9356 |
| AUC-PR (Macro) | 0.9857 |
| Cohen's Kappa | 0.9007 |
| MCC | 0.9010 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.36G |
| FLOPs | 2.72G |
| Params | 21.31M |
| Pure-Inference FPS | 2836.4 (0.35 ms/张) |
| End-to-End FPS | 234.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9429 | 0.9032 | 0.9224 | 0.9666 | 0.9891 | 0.9828 | 208 |
| Malignant_lesions | 0.8920 | 0.9750 | 0.9401 | 0.9155 | 0.9535 | 0.9901 | 0.9778 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9818 | 0.9635 | 0.9635 | 0.9818 | 0.9981 | 0.9964 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=21 TN=347 FN=12
- **Malignant_lesions**: TP=157 FP=10 TN=390 FN=19
- **body_of_stomach_normal**: TP=185 FP=7 TN=377 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 10 | 2 |
| Malignant_lesions | 14 | 157 | 5 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9032    0.9423    0.9224       208
     Malignant_lesions     0.9401    0.8920    0.9155       176
body_of_stomach_normal     0.9635    0.9635    0.9635       192

              accuracy                         0.9340       576
             macro avg     0.9356    0.9326    0.9338       576
          weighted avg     0.9346    0.9340    0.9340       576

```