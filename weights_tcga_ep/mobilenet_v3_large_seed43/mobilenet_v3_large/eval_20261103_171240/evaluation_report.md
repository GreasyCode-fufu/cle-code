# mobilenet_v3_large 评估报告

- 权重: `weights_tcga_ep/mobilenet_v3_large_seed43/mobilenet_v3_large/model_best.pth`
- 测试集: `../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8037 |
| Accuracy | 0.8446 |
| Macro-AUROC (OvR) | 0.9597 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8435 |
| Balanced Acc | 0.8141 |
| Macro-Sensitivity | 0.8141 |
| Macro-Specificity | 0.9680 |
| Macro-Precision | 0.8020 |
| AUC-PR (Macro) | 0.8635 |
| Cohen's Kappa | 0.8002 |
| MCC | 0.8012 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.93M |
| Params | 4.21M |
| Pure-Inference FPS | 9625.0 (0.10 ms/张) |
| End-to-End FPS | 563.3 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.9146 | 0.9928 | 0.7082 | 0.7982 | 0.9984 | 0.9962 | 0.8680 | 199 |
| cancer | 0.9059 | 0.9740 | 0.9201 | 0.9129 | 0.9690 | 0.9899 | 0.9735 | 2645 |
| empty | 0.9963 | 0.9597 | 0.9161 | 0.9545 | 0.9983 | 0.9798 | 0.9157 | 3256 |
| normal | 0.7591 | 0.9876 | 0.9229 | 0.8330 | 0.9546 | 0.9730 | 0.9126 | 1735 |
| other | 0.6721 | 0.9221 | 0.6241 | 0.6472 | 0.9359 | 0.8597 | 0.7326 | 1717 |
| stroma | 0.6368 | 0.9720 | 0.7207 | 0.6762 | 0.9594 | 0.9596 | 0.7788 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=182 FP=75 TN=10360 FN=17
- **cancer**: TP=2396 FP=208 TN=7781 FN=249
- **empty**: TP=3244 FP=297 TN=7081 FN=12
- **normal**: TP=1317 FP=110 TN=8789 FN=418
- **other**: TP=1154 FP=695 TN=8222 FN=563
- **stroma**: TP=689 FP=267 TN=9285 FN=393

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 182 | 3 | 4 | 1 | 7 | 2 |
| cancer | 6 | 2396 | 2 | 50 | 143 | 48 |
| empty | 0 | 3 | 3244 | 0 | 8 | 1 |
| normal | 8 | 52 | 3 | 1317 | 250 | 105 |
| other | 54 | 102 | 284 | 12 | 1154 | 111 |
| stroma | 7 | 48 | 4 | 47 | 287 | 689 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7082    0.9146    0.7982       199
      cancer     0.9201    0.9059    0.9129      2645
       empty     0.9161    0.9963    0.9545      3256
      normal     0.9229    0.7591    0.8330      1735
       other     0.6241    0.6721    0.6472      1717
      stroma     0.7207    0.6368    0.6762      1082

    accuracy                         0.8446     10634
   macro avg     0.8020    0.8141    0.8037     10634
weighted avg     0.8473    0.8446    0.8435     10634

```