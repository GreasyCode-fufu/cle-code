# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/resnet50_seed44/resnet50/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8066 |
| Accuracy | 0.8431 |
| Macro-AUROC (OvR) | 0.9638 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8400 |
| Balanced Acc | 0.8017 |
| Macro-Sensitivity | 0.8017 |
| Macro-Specificity | 0.9675 |
| Macro-Precision | 0.8154 |
| AUC-PR (Macro) | 0.8726 |
| Cohen's Kappa | 0.7977 |
| MCC | 0.7985 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.52M |
| Pure-Inference FPS | 896.8 (1.12 ms/张) |
| End-to-End FPS | 377.5 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8693 | 0.9968 | 0.8398 | 0.8543 | 0.9975 | 0.9980 | 0.9321 | 199 |
| cancer | 0.9335 | 0.9661 | 0.9011 | 0.9170 | 0.9777 | 0.9906 | 0.9758 | 2645 |
| empty | 0.9911 | 0.9612 | 0.9186 | 0.9535 | 0.9959 | 0.9926 | 0.9786 | 3256 |
| normal | 0.7625 | 0.9799 | 0.8808 | 0.8174 | 0.9549 | 0.9546 | 0.8841 | 1735 |
| other | 0.6558 | 0.9280 | 0.6369 | 0.6462 | 0.9333 | 0.9040 | 0.7155 | 1717 |
| stroma | 0.5980 | 0.9730 | 0.7149 | 0.6512 | 0.9553 | 0.9431 | 0.7497 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=173 FP=33 TN=10402 FN=26
- **cancer**: TP=2469 FP=271 TN=7718 FN=176
- **empty**: TP=3227 FP=286 TN=7092 FN=29
- **normal**: TP=1323 FP=179 TN=8720 FN=412
- **other**: TP=1126 FP=642 TN=8275 FN=591
- **stroma**: TP=647 FP=258 TN=9294 FN=435

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 173 | 4 | 4 | 5 | 12 | 1 |
| cancer | 2 | 2469 | 2 | 47 | 79 | 46 |
| empty | 0 | 7 | 3227 | 0 | 22 | 0 |
| normal | 3 | 64 | 0 | 1323 | 241 | 104 |
| other | 24 | 129 | 280 | 51 | 1126 | 107 |
| stroma | 4 | 67 | 0 | 76 | 288 | 647 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8398    0.8693    0.8543       199
      cancer     0.9011    0.9335    0.9170      2645
       empty     0.9186    0.9911    0.9535      3256
      normal     0.8808    0.7625    0.8174      1735
       other     0.6369    0.6558    0.6462      1717
      stroma     0.7149    0.5980    0.6512      1082

    accuracy                         0.8431     10634
   macro avg     0.8154    0.8017    0.8066     10634
weighted avg     0.8404    0.8431    0.8400     10634

```