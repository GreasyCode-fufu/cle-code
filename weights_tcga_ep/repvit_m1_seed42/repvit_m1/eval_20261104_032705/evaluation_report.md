# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/repvit_m1_seed42/repvit_m1/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7567 |
| Accuracy | 0.7930 |
| Macro-AUROC (OvR) | 0.9436 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7948 |
| Balanced Acc | 0.7599 |
| Macro-Sensitivity | 0.7599 |
| Macro-Specificity | 0.9583 |
| Macro-Precision | 0.7676 |
| AUC-PR (Macro) | 0.8176 |
| Cohen's Kappa | 0.7348 |
| MCC | 0.7371 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 2370.6 (0.42 ms/张) |
| End-to-End FPS | 507.5 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8995 | 0.9968 | 0.8443 | 0.8710 | 0.9981 | 0.9979 | 0.9430 | 199 |
| cancer | 0.8612 | 0.9735 | 0.9149 | 0.8872 | 0.9549 | 0.9841 | 0.9602 | 2645 |
| empty | 0.9932 | 0.9603 | 0.9169 | 0.9536 | 0.9969 | 0.9873 | 0.9643 | 3256 |
| normal | 0.5960 | 0.9867 | 0.8976 | 0.7163 | 0.9261 | 0.9024 | 0.8014 | 1735 |
| other | 0.6290 | 0.9101 | 0.5739 | 0.6002 | 0.9272 | 0.8849 | 0.6559 | 1717 |
| stroma | 0.5804 | 0.9222 | 0.4581 | 0.5120 | 0.9510 | 0.9050 | 0.5811 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=179 FP=33 TN=10402 FN=20
- **cancer**: TP=2278 FP=212 TN=7777 FN=367
- **empty**: TP=3234 FP=293 TN=7085 FN=22
- **normal**: TP=1034 FP=118 TN=8781 FN=701
- **other**: TP=1080 FP=802 TN=8115 FN=637
- **stroma**: TP=628 FP=743 TN=8809 FN=454

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 179 | 1 | 1 | 0 | 14 | 4 |
| cancer | 5 | 2278 | 0 | 65 | 152 | 145 |
| empty | 0 | 2 | 3234 | 1 | 16 | 3 |
| normal | 1 | 54 | 1 | 1034 | 295 | 350 |
| other | 23 | 77 | 290 | 6 | 1080 | 241 |
| stroma | 4 | 78 | 1 | 46 | 325 | 628 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8443    0.8995    0.8710       199
      cancer     0.9149    0.8612    0.8872      2645
       empty     0.9169    0.9932    0.9536      3256
      normal     0.8976    0.5960    0.7163      1735
       other     0.5739    0.6290    0.6002      1717
      stroma     0.4581    0.5804    0.5120      1082

    accuracy                         0.7930     10634
   macro avg     0.7676    0.7599    0.7567     10634
weighted avg     0.8098    0.7930    0.7948     10634

```