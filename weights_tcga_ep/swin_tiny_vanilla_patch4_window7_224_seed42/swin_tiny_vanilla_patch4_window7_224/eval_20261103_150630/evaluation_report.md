# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/swin_tiny_vanilla_patch4_window7_224_seed42/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8467 |
| Accuracy | 0.8830 |
| Macro-AUROC (OvR) | 0.9805 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8837 |
| Balanced Acc | 0.8330 |
| Macro-Sensitivity | 0.8330 |
| Macro-Specificity | 0.9765 |
| Macro-Precision | 0.8713 |
| AUC-PR (Macro) | 0.9152 |
| Cohen's Kappa | 0.8522 |
| MCC | 0.8530 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 559.1 (1.79 ms/张) |
| End-to-End FPS | 302.9 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.6600 | 0.9988 | 0.9041 | 0.7630 | 0.9940 | 0.9971 | 0.8956 | 100 |
| cancer | 0.9112 | 0.9616 | 0.8782 | 0.8944 | 0.9727 | 0.9790 | 0.9430 | 1329 |
| empty | 0.9932 | 0.9978 | 0.9944 | 0.9938 | 0.9973 | 0.9998 | 0.9997 | 1621 |
| normal | 0.8442 | 0.9947 | 0.9716 | 0.9034 | 0.9672 | 0.9944 | 0.9774 | 1014 |
| other | 0.7156 | 0.9594 | 0.7624 | 0.7383 | 0.9487 | 0.9401 | 0.8443 | 879 |
| stroma | 0.8735 | 0.9470 | 0.7168 | 0.7874 | 0.9799 | 0.9726 | 0.8312 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=66 FP=7 TN=5595 FN=34
- **cancer**: TP=1211 FP=168 TN=4205 FN=118
- **empty**: TP=1610 FP=9 TN=4072 FN=11
- **normal**: TP=856 FP=25 TN=4663 FN=158
- **other**: TP=629 FP=196 TN=4627 FN=250
- **stroma**: TP=663 FP=262 TN=4681 FN=96

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 66 | 10 | 2 | 5 | 15 | 2 |
| cancer | 6 | 1211 | 0 | 5 | 47 | 60 |
| empty | 0 | 0 | 1610 | 0 | 9 | 2 |
| normal | 1 | 28 | 0 | 856 | 93 | 36 |
| other | 0 | 78 | 7 | 3 | 629 | 162 |
| stroma | 0 | 52 | 0 | 12 | 32 | 663 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.9041    0.6600    0.7630       100
      cancer     0.8782    0.9112    0.8944      1329
       empty     0.9944    0.9932    0.9938      1621
      normal     0.9716    0.8442    0.9034      1014
       other     0.7624    0.7156    0.7383       879
      stroma     0.7168    0.8735    0.7874       759

    accuracy                         0.8830      5702
   macro avg     0.8713    0.8330    0.8467      5702
weighted avg     0.8890    0.8830    0.8837      5702

```