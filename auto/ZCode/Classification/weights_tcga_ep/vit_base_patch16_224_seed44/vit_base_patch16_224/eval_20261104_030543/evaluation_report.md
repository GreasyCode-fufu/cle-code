# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/vit_base_patch16_224_seed44/vit_base_patch16_224/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7903 |
| Accuracy | 0.8202 |
| Macro-AUROC (OvR) | 0.9552 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8193 |
| Balanced Acc | 0.7965 |
| Macro-Sensitivity | 0.7965 |
| Macro-Specificity | 0.9632 |
| Macro-Precision | 0.7875 |
| AUC-PR (Macro) | 0.8517 |
| Cohen's Kappa | 0.7696 |
| MCC | 0.7705 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 360.2 (2.78 ms/张) |
| End-to-End FPS | 231.7 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8693 | 0.9959 | 0.8009 | 0.8337 | 0.9975 | 0.9974 | 0.9033 | 199 |
| cancer | 0.8026 | 0.9733 | 0.9088 | 0.8524 | 0.9371 | 0.9688 | 0.9410 | 2645 |
| empty | 0.9991 | 0.9581 | 0.9133 | 0.9542 | 0.9996 | 0.9858 | 0.9275 | 3256 |
| normal | 0.7337 | 0.9658 | 0.8072 | 0.7687 | 0.9490 | 0.9635 | 0.8648 | 1735 |
| other | 0.6506 | 0.9288 | 0.6376 | 0.6440 | 0.9324 | 0.8628 | 0.7066 | 1717 |
| stroma | 0.7237 | 0.9573 | 0.6574 | 0.6890 | 0.9683 | 0.9528 | 0.7671 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=173 FP=43 TN=10392 FN=26
- **cancer**: TP=2123 FP=213 TN=7776 FN=522
- **empty**: TP=3253 FP=309 TN=7069 FN=3
- **normal**: TP=1273 FP=304 TN=8595 FN=462
- **other**: TP=1117 FP=635 TN=8282 FN=600
- **stroma**: TP=783 FP=408 TN=9144 FN=299

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 173 | 2 | 2 | 4 | 17 | 1 |
| cancer | 4 | 2123 | 3 | 169 | 251 | 95 |
| empty | 0 | 0 | 3253 | 0 | 3 | 0 |
| normal | 2 | 107 | 2 | 1273 | 203 | 148 |
| other | 36 | 57 | 302 | 41 | 1117 | 164 |
| stroma | 1 | 47 | 0 | 90 | 161 | 783 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8009    0.8693    0.8337       199
      cancer     0.9088    0.8026    0.8524      2645
       empty     0.9133    0.9991    0.9542      3256
      normal     0.8072    0.7337    0.7687      1735
       other     0.6376    0.6506    0.6440      1717
      stroma     0.6574    0.7237    0.6890      1082

    accuracy                         0.8202     10634
   macro avg     0.7875    0.7965    0.7903     10634
weighted avg     0.8222    0.8202    0.8193     10634

```