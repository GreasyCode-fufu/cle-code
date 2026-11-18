# convnext_tiny_cle_oca_strip 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_tiny_cle_oca_strip_seed43/convnext_tiny_cle_oca_strip/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8189 |
| Accuracy | 0.8512 |
| Macro-AUROC (OvR) | 0.9607 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8475 |
| Balanced Acc | 0.8210 |
| Macro-Sensitivity | 0.8210 |
| Macro-Specificity | 0.9689 |
| Macro-Precision | 0.8224 |
| AUC-PR (Macro) | 0.8846 |
| Cohen's Kappa | 0.8084 |
| MCC | 0.8096 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.47G |
| FLOPs | 8.94G |
| Params | 27.84M |
| Pure-Inference FPS | 758.0 (1.32 ms/张) |
| End-to-End FPS | 355.1 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8593 | 0.9955 | 0.7844 | 0.8201 | 0.9973 | 0.9965 | 0.9167 | 199 |
| cancer | 0.9081 | 0.9432 | 0.8410 | 0.8733 | 0.9688 | 0.9803 | 0.9467 | 2645 |
| empty | 0.9954 | 0.9619 | 0.9202 | 0.9563 | 0.9979 | 0.9816 | 0.9506 | 3256 |
| normal | 0.7620 | 0.9838 | 0.9018 | 0.8260 | 0.9550 | 0.9742 | 0.9109 | 1735 |
| other | 0.6302 | 0.9611 | 0.7572 | 0.6879 | 0.9310 | 0.8656 | 0.7585 | 1717 |
| stroma | 0.7708 | 0.9677 | 0.7297 | 0.7497 | 0.9739 | 0.9662 | 0.8241 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=171 FP=47 TN=10388 FN=28
- **cancer**: TP=2402 FP=454 TN=7535 FN=243
- **empty**: TP=3241 FP=281 TN=7097 FN=15
- **normal**: TP=1322 FP=144 TN=8755 FN=413
- **other**: TP=1082 FP=347 TN=8570 FN=635
- **stroma**: TP=834 FP=309 TN=9243 FN=248

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 171 | 5 | 2 | 3 | 16 | 2 |
| cancer | 5 | 2402 | 2 | 54 | 128 | 54 |
| empty | 0 | 2 | 3241 | 0 | 13 | 0 |
| normal | 3 | 176 | 2 | 1322 | 102 | 130 |
| other | 37 | 172 | 275 | 28 | 1082 | 123 |
| stroma | 2 | 99 | 0 | 59 | 88 | 834 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7844    0.8593    0.8201       199
      cancer     0.8410    0.9081    0.8733      2645
       empty     0.9202    0.9954    0.9563      3256
      normal     0.9018    0.7620    0.8260      1735
       other     0.7572    0.6302    0.6879      1717
      stroma     0.7297    0.7708    0.7497      1082

    accuracy                         0.8512     10634
   macro avg     0.8224    0.8210    0.8189     10634
weighted avg     0.8493    0.8512    0.8475     10634

```