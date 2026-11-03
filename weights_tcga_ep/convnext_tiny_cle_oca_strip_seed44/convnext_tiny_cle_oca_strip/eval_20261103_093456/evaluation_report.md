# convnext_tiny_cle_oca_strip 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_tiny_cle_oca_strip_seed44/convnext_tiny_cle_oca_strip/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8118 |
| Accuracy | 0.8469 |
| Macro-AUROC (OvR) | 0.9648 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8405 |
| Balanced Acc | 0.7925 |
| Macro-Sensitivity | 0.7925 |
| Macro-Specificity | 0.9675 |
| Macro-Precision | 0.8405 |
| AUC-PR (Macro) | 0.8710 |
| Cohen's Kappa | 0.8017 |
| MCC | 0.8038 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.47G |
| FLOPs | 8.94G |
| Params | 27.84M |
| Pure-Inference FPS | 1562.2 (0.64 ms/张) |
| End-to-End FPS | 468.9 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7789 | 0.9988 | 0.9226 | 0.8447 | 0.9958 | 0.9848 | 0.9230 | 199 |
| cancer | 0.9471 | 0.9293 | 0.8160 | 0.8766 | 0.9815 | 0.9834 | 0.9564 | 2645 |
| empty | 0.9951 | 0.9588 | 0.9142 | 0.9529 | 0.9977 | 0.9799 | 0.9410 | 3256 |
| normal | 0.8127 | 0.9817 | 0.8964 | 0.8525 | 0.9641 | 0.9781 | 0.9234 | 1735 |
| other | 0.5894 | 0.9588 | 0.7339 | 0.6537 | 0.9238 | 0.9107 | 0.7291 | 1717 |
| stroma | 0.6322 | 0.9774 | 0.7600 | 0.6902 | 0.9591 | 0.9518 | 0.7529 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=155 FP=13 TN=10422 FN=44
- **cancer**: TP=2505 FP=565 TN=7424 FN=140
- **empty**: TP=3240 FP=304 TN=7074 FN=16
- **normal**: TP=1410 FP=163 TN=8736 FN=325
- **other**: TP=1012 FP=367 TN=8550 FN=705
- **stroma**: TP=684 FP=216 TN=9336 FN=398

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 155 | 12 | 3 | 7 | 21 | 1 |
| cancer | 1 | 2505 | 6 | 23 | 73 | 37 |
| empty | 0 | 4 | 3240 | 0 | 12 | 0 |
| normal | 0 | 181 | 0 | 1410 | 82 | 62 |
| other | 11 | 262 | 295 | 21 | 1012 | 116 |
| stroma | 1 | 106 | 0 | 112 | 179 | 684 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.9226    0.7789    0.8447       199
      cancer     0.8160    0.9471    0.8766      2645
       empty     0.9142    0.9951    0.9529      3256
      normal     0.8964    0.8127    0.8525      1735
       other     0.7339    0.5894    0.6537      1717
      stroma     0.7600    0.6322    0.6902      1082

    accuracy                         0.8469     10634
   macro avg     0.8405    0.7925    0.8118     10634
weighted avg     0.8422    0.8469    0.8405     10634

```