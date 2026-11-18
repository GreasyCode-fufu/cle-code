# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/swin_tiny_vanilla_patch4_window7_224_seed42/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7895 |
| Accuracy | 0.8213 |
| Macro-AUROC (OvR) | 0.9594 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8168 |
| Balanced Acc | 0.7837 |
| Macro-Sensitivity | 0.7837 |
| Macro-Specificity | 0.9627 |
| Macro-Precision | 0.8075 |
| AUC-PR (Macro) | 0.8587 |
| Cohen's Kappa | 0.7693 |
| MCC | 0.7718 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 568.6 (1.76 ms/张) |
| End-to-End FPS | 305.4 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8593 | 0.9973 | 0.8593 | 0.8593 | 0.9973 | 0.9941 | 0.9180 | 199 |
| cancer | 0.9195 | 0.9234 | 0.7989 | 0.8550 | 0.9719 | 0.9704 | 0.9282 | 2645 |
| empty | 0.9942 | 0.9631 | 0.9225 | 0.9570 | 0.9973 | 0.9842 | 0.9617 | 3256 |
| normal | 0.6403 | 0.9936 | 0.9512 | 0.7654 | 0.9341 | 0.9623 | 0.8932 | 1735 |
| other | 0.6115 | 0.9389 | 0.6583 | 0.6341 | 0.9262 | 0.8928 | 0.7109 | 1717 |
| stroma | 0.6774 | 0.9596 | 0.6550 | 0.6661 | 0.9633 | 0.9525 | 0.7404 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=171 FP=28 TN=10407 FN=28
- **cancer**: TP=2432 FP=612 TN=7377 FN=213
- **empty**: TP=3237 FP=272 TN=7106 FN=19
- **normal**: TP=1111 FP=57 TN=8842 FN=624
- **other**: TP=1050 FP=545 TN=8372 FN=667
- **stroma**: TP=733 FP=386 TN=9166 FN=349

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 171 | 10 | 2 | 5 | 10 | 1 |
| cancer | 1 | 2432 | 0 | 10 | 121 | 81 |
| empty | 0 | 0 | 3237 | 0 | 19 | 0 |
| normal | 3 | 263 | 0 | 1111 | 216 | 142 |
| other | 23 | 204 | 270 | 8 | 1050 | 162 |
| stroma | 1 | 135 | 0 | 34 | 179 | 733 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8593    0.8593    0.8593       199
      cancer     0.7989    0.9195    0.8550      2645
       empty     0.9225    0.9942    0.9570      3256
      normal     0.9512    0.6403    0.7654      1735
       other     0.6583    0.6115    0.6341      1717
      stroma     0.6550    0.6774    0.6661      1082

    accuracy                         0.8213     10634
   macro avg     0.8075    0.7837    0.7895     10634
weighted avg     0.8254    0.8213    0.8168     10634

```