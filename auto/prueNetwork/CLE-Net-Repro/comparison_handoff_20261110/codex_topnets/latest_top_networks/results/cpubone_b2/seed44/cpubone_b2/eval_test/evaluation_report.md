# cpubone_b2 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/cpubone_b2/seed44/cpubone_b2/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9162 |
| Accuracy | 0.9184 |
| Macro-AUROC (OvR) | 0.9890 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9174 |
| Balanced Acc | 0.9146 |
| Macro-Sensitivity | 0.9146 |
| Macro-Specificity | 0.9585 |
| Macro-Precision | 0.9222 |
| AUC-PR (Macro) | 0.9786 |
| Cohen's Kappa | 0.8770 |
| MCC | 0.8788 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.36G |
| FLOPs | 2.72G |
| Params | 21.31M |
| Pure-Inference FPS | 2827.8 (0.35 ms/张) |
| End-to-End FPS | 248.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9293 | 0.8844 | 0.9192 | 0.9744 | 0.9864 | 0.9788 | 208 |
| Malignant_lesions | 0.8182 | 0.9800 | 0.9474 | 0.8780 | 0.9245 | 0.9835 | 0.9621 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9661 | 0.9347 | 0.9514 | 0.9841 | 0.9972 | 0.9948 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=26 TN=342 FN=9
- **Malignant_lesions**: TP=144 FP=8 TN=392 FN=32
- **body_of_stomach_normal**: TP=186 FP=13 TN=371 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 6 | 3 |
| Malignant_lesions | 22 | 144 | 10 |
| body_of_stomach_normal | 4 | 2 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8844    0.9567    0.9192       208
     Malignant_lesions     0.9474    0.8182    0.8780       176
body_of_stomach_normal     0.9347    0.9688    0.9514       192

              accuracy                         0.9184       576
             macro avg     0.9222    0.9146    0.9162       576
          weighted avg     0.9204    0.9184    0.9174       576

```