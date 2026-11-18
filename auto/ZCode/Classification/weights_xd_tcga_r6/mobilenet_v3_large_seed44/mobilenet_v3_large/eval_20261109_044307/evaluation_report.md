# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mobilenet_v3_large_seed44/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7968 |
| Accuracy | 0.8416 |
| Macro-AUROC (OvR) | 0.9501 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8373 |
| Balanced Acc | 0.8116 |
| Macro-Sensitivity | 0.8116 |
| Macro-Specificity | 0.9672 |
| Macro-Precision | 0.7923 |
| AUC-PR (Macro) | 0.8500 |
| Cohen's Kappa | 0.7960 |
| MCC | 0.7969 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.93M |
| Params | 4.21M |
| Pure-Inference FPS | 6853.7 (0.15 ms/张) |
| End-to-End FPS | 200.9 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.9447 | 0.9920 | 0.6912 | 0.7983 | 0.9989 | 0.9968 | 0.9207 | 199 |
| cancer | 0.9289 | 0.9568 | 0.8769 | 0.9021 | 0.9760 | 0.9898 | 0.9723 | 2645 |
| empty | 0.9957 | 0.9610 | 0.9184 | 0.9555 | 0.9980 | 0.9704 | 0.8604 | 3256 |
| normal | 0.7637 | 0.9812 | 0.8881 | 0.8212 | 0.9552 | 0.9500 | 0.8822 | 1735 |
| other | 0.6296 | 0.9407 | 0.6714 | 0.6498 | 0.9295 | 0.8427 | 0.7163 | 1717 |
| stroma | 0.6072 | 0.9716 | 0.7080 | 0.6537 | 0.9562 | 0.9508 | 0.7480 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=188 FP=84 TN=10351 FN=11
- **cancer**: TP=2457 FP=345 TN=7644 FN=188
- **empty**: TP=3242 FP=288 TN=7090 FN=14
- **normal**: TP=1325 FP=167 TN=8732 FN=410
- **other**: TP=1081 FP=529 TN=8388 FN=636
- **stroma**: TP=657 FP=271 TN=9281 FN=425

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 188 | 3 | 2 | 2 | 3 | 1 |
| cancer | 5 | 2457 | 1 | 65 | 88 | 29 |
| empty | 0 | 8 | 3242 | 0 | 6 | 0 |
| normal | 4 | 143 | 3 | 1325 | 172 | 88 |
| other | 65 | 116 | 282 | 20 | 1081 | 153 |
| stroma | 10 | 75 | 0 | 80 | 260 | 657 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6912    0.9447    0.7983       199
      cancer     0.8769    0.9289    0.9021      2645
       empty     0.9184    0.9957    0.9555      3256
      normal     0.8881    0.7637    0.8212      1735
       other     0.6714    0.6296    0.6498      1717
      stroma     0.7080    0.6072    0.6537      1082

    accuracy                         0.8416     10634
   macro avg     0.7923    0.8116    0.7968     10634
weighted avg     0.8376    0.8416    0.8373     10634

```