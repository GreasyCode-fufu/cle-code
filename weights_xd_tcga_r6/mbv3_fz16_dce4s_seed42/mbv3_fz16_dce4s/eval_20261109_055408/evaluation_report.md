# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce4s_seed42/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7956 |
| Accuracy | 0.8404 |
| Macro-AUROC (OvR) | 0.9614 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8362 |
| Balanced Acc | 0.7983 |
| Macro-Sensitivity | 0.7983 |
| Macro-Specificity | 0.9669 |
| Macro-Precision | 0.7985 |
| AUC-PR (Macro) | 0.8642 |
| Cohen's Kappa | 0.7943 |
| MCC | 0.7956 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9588.8 (0.10 ms/张) |
| End-to-End FPS | 546.8 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8291 | 0.9943 | 0.7366 | 0.7801 | 0.9967 | 0.9928 | 0.8851 | 199 |
| cancer | 0.9263 | 0.9493 | 0.8581 | 0.8909 | 0.9749 | 0.9882 | 0.9663 | 2645 |
| empty | 0.9966 | 0.9606 | 0.9177 | 0.9555 | 0.9985 | 0.9878 | 0.9605 | 3256 |
| normal | 0.7262 | 0.9815 | 0.8842 | 0.7975 | 0.9484 | 0.9509 | 0.8735 | 1735 |
| other | 0.6267 | 0.9529 | 0.7193 | 0.6698 | 0.9299 | 0.8903 | 0.7432 | 1717 |
| stroma | 0.6848 | 0.9626 | 0.6749 | 0.6798 | 0.9642 | 0.9584 | 0.7563 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=165 FP=59 TN=10376 FN=34
- **cancer**: TP=2450 FP=405 TN=7584 FN=195
- **empty**: TP=3245 FP=291 TN=7087 FN=11
- **normal**: TP=1260 FP=165 TN=8734 FN=475
- **other**: TP=1076 FP=420 TN=8497 FN=641
- **stroma**: TP=741 FP=357 TN=9195 FN=341

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 165 | 3 | 2 | 8 | 17 | 4 |
| cancer | 8 | 2450 | 1 | 45 | 87 | 54 |
| empty | 0 | 0 | 3245 | 0 | 11 | 0 |
| normal | 4 | 175 | 3 | 1260 | 146 | 147 |
| other | 42 | 142 | 285 | 20 | 1076 | 152 |
| stroma | 5 | 85 | 0 | 92 | 159 | 741 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7366    0.8291    0.7801       199
      cancer     0.8581    0.9263    0.8909      2645
       empty     0.9177    0.9966    0.9555      3256
      normal     0.8842    0.7262    0.7975      1735
       other     0.7193    0.6267    0.6698      1717
      stroma     0.6749    0.6848    0.6798      1082

    accuracy                         0.8404     10634
   macro avg     0.7985    0.7983    0.7956     10634
weighted avg     0.8373    0.8404    0.8362     10634

```