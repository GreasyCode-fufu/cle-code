# convnext_tiny_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_dce_last_seed42/convnext_tiny_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9423 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9941 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9425 |
| Balanced Acc | 0.9419 |
| Macro-Sensitivity | 0.9419 |
| Macro-Specificity | 0.9711 |
| Macro-Precision | 0.9430 |
| AUC-PR (Macro) | 0.9879 |
| Cohen's Kappa | 0.9138 |
| MCC | 0.9139 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 951.7 (1.05 ms/张) |
| End-to-End FPS | 11.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9592 | 0.9286 | 0.9330 | 0.9645 | 0.9926 | 0.9876 | 208 |
| Malignant_lesions | 0.9091 | 0.9750 | 0.9412 | 0.9249 | 0.9606 | 0.9918 | 0.9797 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9792 | 0.9592 | 0.9691 | 0.9895 | 0.9978 | 0.9962 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=15 TN=353 FN=13
- **Malignant_lesions**: TP=160 FP=10 TN=390 FN=16
- **body_of_stomach_normal**: TP=188 FP=8 TN=376 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 10 | 3 |
| Malignant_lesions | 11 | 160 | 5 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9286    0.9375    0.9330       208
     Malignant_lesions     0.9412    0.9091    0.9249       176
body_of_stomach_normal     0.9592    0.9792    0.9691       192

              accuracy                         0.9427       576
             macro avg     0.9430    0.9419    0.9423       576
          weighted avg     0.9426    0.9427    0.9425       576

```