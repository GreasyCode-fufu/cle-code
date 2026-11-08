# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce16_seed42/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7923 |
| Accuracy | 0.8382 |
| Macro-AUROC (OvR) | 0.9605 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8341 |
| Balanced Acc | 0.7944 |
| Macro-Sensitivity | 0.7944 |
| Macro-Specificity | 0.9664 |
| Macro-Precision | 0.7955 |
| AUC-PR (Macro) | 0.8607 |
| Cohen's Kappa | 0.7914 |
| MCC | 0.7926 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9619.4 (0.10 ms/张) |
| End-to-End FPS | 550.2 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8141 | 0.9942 | 0.7265 | 0.7678 | 0.9964 | 0.9920 | 0.8716 | 199 |
| cancer | 0.9229 | 0.9481 | 0.8547 | 0.8875 | 0.9738 | 0.9877 | 0.9652 | 2645 |
| empty | 0.9939 | 0.9606 | 0.9175 | 0.9542 | 0.9972 | 0.9887 | 0.9633 | 3256 |
| normal | 0.7228 | 0.9819 | 0.8862 | 0.7962 | 0.9478 | 0.9479 | 0.8716 | 1735 |
| other | 0.6296 | 0.9502 | 0.7089 | 0.6669 | 0.9302 | 0.8888 | 0.7371 | 1717 |
| stroma | 0.6830 | 0.9635 | 0.6792 | 0.6811 | 0.9641 | 0.9576 | 0.7553 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=162 FP=61 TN=10374 FN=37
- **cancer**: TP=2441 FP=415 TN=7574 FN=204
- **empty**: TP=3236 FP=291 TN=7087 FN=20
- **normal**: TP=1254 FP=161 TN=8738 FN=481
- **other**: TP=1081 FP=444 TN=8473 FN=636
- **stroma**: TP=739 FP=349 TN=9203 FN=343

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 162 | 3 | 2 | 7 | 21 | 4 |
| cancer | 8 | 2441 | 1 | 43 | 95 | 57 |
| empty | 0 | 2 | 3236 | 0 | 18 | 0 |
| normal | 6 | 181 | 4 | 1254 | 149 | 141 |
| other | 41 | 143 | 284 | 21 | 1081 | 147 |
| stroma | 6 | 86 | 0 | 90 | 161 | 739 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7265    0.8141    0.7678       199
      cancer     0.8547    0.9229    0.8875      2645
       empty     0.9175    0.9939    0.9542      3256
      normal     0.8862    0.7228    0.7962      1735
       other     0.7089    0.6296    0.6669      1717
      stroma     0.6792    0.6830    0.6811      1082

    accuracy                         0.8382     10634
   macro avg     0.7955    0.7944    0.7923     10634
weighted avg     0.8353    0.8382    0.8341     10634

```