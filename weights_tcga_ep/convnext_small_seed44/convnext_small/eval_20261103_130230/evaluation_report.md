# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_small_seed44/convnext_small/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8045 |
| Accuracy | 0.8432 |
| Macro-AUROC (OvR) | 0.9629 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8386 |
| Balanced Acc | 0.8000 |
| Macro-Sensitivity | 0.8000 |
| Macro-Specificity | 0.9673 |
| Macro-Precision | 0.8201 |
| AUC-PR (Macro) | 0.8643 |
| Cohen's Kappa | 0.7979 |
| MCC | 0.8003 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.42M |
| Pure-Inference FPS | 387.1 (2.58 ms/张) |
| End-to-End FPS | 245.1 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7839 | 0.9967 | 0.8211 | 0.8021 | 0.9959 | 0.9966 | 0.9026 | 199 |
| cancer | 0.9259 | 0.9339 | 0.8226 | 0.8712 | 0.9744 | 0.9763 | 0.9313 | 2645 |
| empty | 0.9948 | 0.9630 | 0.9223 | 0.9572 | 0.9976 | 0.9800 | 0.9406 | 3256 |
| normal | 0.7458 | 0.9870 | 0.9177 | 0.8229 | 0.9522 | 0.9742 | 0.9129 | 1735 |
| other | 0.5807 | 0.9689 | 0.7826 | 0.6667 | 0.9231 | 0.8989 | 0.7533 | 1717 |
| stroma | 0.7689 | 0.9540 | 0.6546 | 0.7072 | 0.9733 | 0.9512 | 0.7451 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=156 FP=34 TN=10401 FN=43
- **cancer**: TP=2449 FP=528 TN=7461 FN=196
- **empty**: TP=3239 FP=273 TN=7105 FN=17
- **normal**: TP=1294 FP=116 TN=8783 FN=441
- **other**: TP=997 FP=277 TN=8640 FN=720
- **stroma**: TP=832 FP=439 TN=9113 FN=250

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 156 | 4 | 2 | 6 | 18 | 13 |
| cancer | 2 | 2449 | 1 | 45 | 81 | 67 |
| empty | 0 | 0 | 3239 | 0 | 17 | 0 |
| normal | 1 | 199 | 0 | 1294 | 95 | 146 |
| other | 28 | 190 | 270 | 19 | 997 | 213 |
| stroma | 3 | 135 | 0 | 46 | 66 | 832 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8211    0.7839    0.8021       199
      cancer     0.8226    0.9259    0.8712      2645
       empty     0.9223    0.9948    0.9572      3256
      normal     0.9177    0.7458    0.8229      1735
       other     0.7826    0.5807    0.6667      1717
      stroma     0.6546    0.7689    0.7072      1082

    accuracy                         0.8432     10634
   macro avg     0.8201    0.8000    0.8045     10634
weighted avg     0.8451    0.8432    0.8386     10634

```