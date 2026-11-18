# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_egb_scg_seed44/convnext_egb_scg/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8057 |
| Accuracy | 0.8425 |
| Macro-AUROC (OvR) | 0.9576 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8405 |
| Balanced Acc | 0.8142 |
| Macro-Sensitivity | 0.8142 |
| Macro-Specificity | 0.9677 |
| Macro-Precision | 0.8002 |
| AUC-PR (Macro) | 0.8583 |
| Cohen's Kappa | 0.7978 |
| MCC | 0.7983 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.29M |
| Pure-Inference FPS | 739.6 (1.35 ms/张) |
| End-to-End FPS | 352.2 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8693 | 0.9940 | 0.7331 | 0.7954 | 0.9975 | 0.9968 | 0.9020 | 199 |
| cancer | 0.8647 | 0.9767 | 0.9248 | 0.8937 | 0.9561 | 0.9849 | 0.9630 | 2645 |
| empty | 0.9972 | 0.9615 | 0.9196 | 0.9568 | 0.9987 | 0.9745 | 0.9096 | 3256 |
| normal | 0.7908 | 0.9609 | 0.7977 | 0.7942 | 0.9593 | 0.9546 | 0.8786 | 1735 |
| other | 0.6377 | 0.9392 | 0.6689 | 0.6530 | 0.9309 | 0.8695 | 0.7020 | 1717 |
| stroma | 0.7255 | 0.9736 | 0.7570 | 0.7409 | 0.9691 | 0.9652 | 0.7948 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=173 FP=63 TN=10372 FN=26
- **cancer**: TP=2287 FP=186 TN=7803 FN=358
- **empty**: TP=3247 FP=284 TN=7094 FN=9
- **normal**: TP=1372 FP=348 TN=8551 FN=363
- **other**: TP=1095 FP=542 TN=8375 FN=622
- **stroma**: TP=785 FP=252 TN=9300 FN=297

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 173 | 5 | 2 | 6 | 11 | 2 |
| cancer | 6 | 2287 | 2 | 205 | 117 | 28 |
| empty | 0 | 0 | 3247 | 0 | 9 | 0 |
| normal | 2 | 48 | 0 | 1372 | 238 | 75 |
| other | 50 | 100 | 280 | 45 | 1095 | 147 |
| stroma | 5 | 33 | 0 | 92 | 167 | 785 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7331    0.8693    0.7954       199
      cancer     0.9248    0.8647    0.8937      2645
       empty     0.9196    0.9972    0.9568      3256
      normal     0.7977    0.7908    0.7942      1735
       other     0.6689    0.6377    0.6530      1717
      stroma     0.7570    0.7255    0.7409      1082

    accuracy                         0.8425     10634
   macro avg     0.8002    0.8142    0.8057     10634
weighted avg     0.8405    0.8425    0.8405     10634

```