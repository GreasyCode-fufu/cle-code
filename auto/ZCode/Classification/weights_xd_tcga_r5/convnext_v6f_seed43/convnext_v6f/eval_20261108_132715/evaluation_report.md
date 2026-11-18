# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r5/convnext_v6f_seed43/convnext_v6f/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8145 |
| Accuracy | 0.8490 |
| Macro-AUROC (OvR) | 0.9674 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8459 |
| Balanced Acc | 0.8109 |
| Macro-Sensitivity | 0.8109 |
| Macro-Specificity | 0.9686 |
| Macro-Precision | 0.8222 |
| AUC-PR (Macro) | 0.8792 |
| Cohen's Kappa | 0.8053 |
| MCC | 0.8062 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.26M |
| Pure-Inference FPS | 615.8 (1.62 ms/张) |
| End-to-End FPS | 318.9 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8693 | 0.9964 | 0.8199 | 0.8439 | 0.9975 | 0.9973 | 0.9265 | 199 |
| cancer | 0.9331 | 0.9568 | 0.8774 | 0.9044 | 0.9774 | 0.9863 | 0.9682 | 2645 |
| empty | 0.9932 | 0.9629 | 0.9219 | 0.9562 | 0.9969 | 0.9813 | 0.9484 | 3256 |
| normal | 0.7608 | 0.9854 | 0.9103 | 0.8289 | 0.9548 | 0.9656 | 0.8996 | 1735 |
| other | 0.6564 | 0.9360 | 0.6637 | 0.6600 | 0.9340 | 0.9139 | 0.7378 | 1717 |
| stroma | 0.6525 | 0.9740 | 0.7400 | 0.6935 | 0.9612 | 0.9603 | 0.7947 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=173 FP=38 TN=10397 FN=26
- **cancer**: TP=2468 FP=345 TN=7644 FN=177
- **empty**: TP=3234 FP=274 TN=7104 FN=22
- **normal**: TP=1320 FP=130 TN=8769 FN=415
- **other**: TP=1127 FP=571 TN=8346 FN=590
- **stroma**: TP=706 FP=248 TN=9304 FN=376

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 173 | 6 | 2 | 5 | 12 | 1 |
| cancer | 3 | 2468 | 2 | 39 | 89 | 44 |
| empty | 0 | 2 | 3234 | 0 | 20 | 0 |
| normal | 1 | 121 | 0 | 1320 | 210 | 83 |
| other | 31 | 151 | 270 | 18 | 1127 | 120 |
| stroma | 3 | 65 | 0 | 68 | 240 | 706 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8199    0.8693    0.8439       199
      cancer     0.8774    0.9331    0.9044      2645
       empty     0.9219    0.9932    0.9562      3256
      normal     0.9103    0.7608    0.8289      1735
       other     0.6637    0.6564    0.6600      1717
      stroma     0.7400    0.6525    0.6935      1082

    accuracy                         0.8490     10634
   macro avg     0.8222    0.8109    0.8145     10634
weighted avg     0.8468    0.8490    0.8459     10634

```