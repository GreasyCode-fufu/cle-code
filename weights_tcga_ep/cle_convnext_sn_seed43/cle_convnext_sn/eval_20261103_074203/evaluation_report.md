# cle_convnext_sn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/cle_convnext_sn_seed43/cle_convnext_sn/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8065 |
| Accuracy | 0.8342 |
| Macro-AUROC (OvR) | 0.9573 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8302 |
| Balanced Acc | 0.8012 |
| Macro-Sensitivity | 0.8012 |
| Macro-Specificity | 0.9655 |
| Macro-Precision | 0.8159 |
| AUC-PR (Macro) | 0.8601 |
| Cohen's Kappa | 0.7866 |
| MCC | 0.7875 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 607.2 (1.65 ms/张) |
| End-to-End FPS | 316.2 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8342 | 0.9979 | 0.8830 | 0.8579 | 0.9968 | 0.9971 | 0.9326 | 199 |
| cancer | 0.8737 | 0.9448 | 0.8398 | 0.8564 | 0.9576 | 0.9723 | 0.9275 | 2645 |
| empty | 0.9923 | 0.9638 | 0.9237 | 0.9568 | 0.9965 | 0.9765 | 0.9085 | 3256 |
| normal | 0.7625 | 0.9560 | 0.7714 | 0.7670 | 0.9538 | 0.9451 | 0.8504 | 1735 |
| other | 0.6069 | 0.9651 | 0.7701 | 0.6788 | 0.9273 | 0.8970 | 0.7627 | 1717 |
| stroma | 0.7375 | 0.9655 | 0.7074 | 0.7222 | 0.9701 | 0.9558 | 0.7790 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=166 FP=22 TN=10413 FN=33
- **cancer**: TP=2311 FP=441 TN=7548 FN=334
- **empty**: TP=3231 FP=267 TN=7111 FN=25
- **normal**: TP=1323 FP=392 TN=8507 FN=412
- **other**: TP=1042 FP=311 TN=8606 FN=675
- **stroma**: TP=798 FP=330 TN=9222 FN=284

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 166 | 6 | 2 | 6 | 16 | 3 |
| cancer | 2 | 2311 | 0 | 219 | 78 | 35 |
| empty | 0 | 2 | 3231 | 0 | 22 | 1 |
| normal | 1 | 151 | 0 | 1323 | 114 | 146 |
| other | 17 | 198 | 265 | 50 | 1042 | 145 |
| stroma | 2 | 84 | 0 | 117 | 81 | 798 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8830    0.8342    0.8579       199
      cancer     0.8398    0.8737    0.8564      2645
       empty     0.9237    0.9923    0.9568      3256
      normal     0.7714    0.7625    0.7670      1735
       other     0.7701    0.6069    0.6788      1717
      stroma     0.7074    0.7375    0.7222      1082

    accuracy                         0.8342     10634
   macro avg     0.8159    0.8012    0.8065     10634
weighted avg     0.8304    0.8342    0.8302     10634

```