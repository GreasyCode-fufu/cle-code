# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/fast_t0_seed44/fast_t0/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8571 |
| Accuracy | 0.8832 |
| Macro-AUROC (OvR) | 0.9774 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8848 |
| Balanced Acc | 0.8655 |
| Macro-Sensitivity | 0.8655 |
| Macro-Specificity | 0.9769 |
| Macro-Precision | 0.8540 |
| AUC-PR (Macro) | 0.9095 |
| Cohen's Kappa | 0.8529 |
| MCC | 0.8540 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 12180.4 (0.08 ms/张) |
| End-to-End FPS | 593.1 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8500 | 0.9963 | 0.8019 | 0.8252 | 0.9973 | 0.9981 | 0.9260 | 100 |
| cancer | 0.8473 | 0.9824 | 0.9360 | 0.8894 | 0.9549 | 0.9860 | 0.9606 | 1329 |
| empty | 0.9932 | 0.9971 | 0.9926 | 0.9929 | 0.9973 | 0.9999 | 0.9999 | 1621 |
| normal | 0.8964 | 0.9855 | 0.9304 | 0.9131 | 0.9778 | 0.9918 | 0.9599 | 1014 |
| other | 0.7258 | 0.9614 | 0.7743 | 0.7493 | 0.9506 | 0.9227 | 0.8255 | 879 |
| stroma | 0.8801 | 0.9389 | 0.6887 | 0.7727 | 0.9808 | 0.9657 | 0.7849 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=85 FP=21 TN=5581 FN=15
- **cancer**: TP=1126 FP=77 TN=4296 FN=203
- **empty**: TP=1610 FP=12 TN=4069 FN=11
- **normal**: TP=909 FP=68 TN=4620 FN=105
- **other**: TP=638 FP=186 TN=4637 FN=241
- **stroma**: TP=668 FP=302 TN=4641 FN=91

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 85 | 2 | 1 | 3 | 3 | 6 |
| cancer | 9 | 1126 | 1 | 30 | 73 | 90 |
| empty | 1 | 0 | 1610 | 0 | 8 | 2 |
| normal | 4 | 12 | 0 | 909 | 59 | 30 |
| other | 5 | 42 | 10 | 10 | 638 | 174 |
| stroma | 2 | 21 | 0 | 25 | 43 | 668 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8019    0.8500    0.8252       100
      cancer     0.9360    0.8473    0.8894      1329
       empty     0.9926    0.9932    0.9929      1621
      normal     0.9304    0.8964    0.9131      1014
       other     0.7743    0.7258    0.7493       879
      stroma     0.6887    0.8801    0.7727       759

    accuracy                         0.8832      5702
   macro avg     0.8540    0.8655    0.8571      5702
weighted avg     0.8909    0.8832    0.8848      5702

```