# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce4m_seed44/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7914 |
| Accuracy | 0.8347 |
| Macro-AUROC (OvR) | 0.9516 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8307 |
| Balanced Acc | 0.7903 |
| Macro-Sensitivity | 0.7903 |
| Macro-Specificity | 0.9657 |
| Macro-Precision | 0.7978 |
| AUC-PR (Macro) | 0.8473 |
| Cohen's Kappa | 0.7867 |
| MCC | 0.7877 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9222.0 (0.11 ms/张) |
| End-to-End FPS | 551.7 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8593 | 0.9953 | 0.7773 | 0.8162 | 0.9973 | 0.9960 | 0.9016 | 199 |
| cancer | 0.9229 | 0.9544 | 0.8702 | 0.8958 | 0.9739 | 0.9871 | 0.9647 | 2645 |
| empty | 0.9954 | 0.9602 | 0.9168 | 0.9545 | 0.9979 | 0.9878 | 0.9638 | 3256 |
| normal | 0.7516 | 0.9835 | 0.8987 | 0.8186 | 0.9531 | 0.9449 | 0.8796 | 1735 |
| other | 0.6407 | 0.9308 | 0.6407 | 0.6407 | 0.9308 | 0.8689 | 0.6771 | 1717 |
| stroma | 0.5721 | 0.9700 | 0.6832 | 0.6227 | 0.9524 | 0.9251 | 0.6967 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=171 FP=49 TN=10386 FN=28
- **cancer**: TP=2441 FP=364 TN=7625 FN=204
- **empty**: TP=3241 FP=294 TN=7084 FN=15
- **normal**: TP=1304 FP=147 TN=8752 FN=431
- **other**: TP=1100 FP=617 TN=8300 FN=617
- **stroma**: TP=619 FP=287 TN=9265 FN=463

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 171 | 5 | 2 | 6 | 12 | 3 |
| cancer | 6 | 2441 | 1 | 40 | 107 | 50 |
| empty | 0 | 0 | 3241 | 0 | 15 | 0 |
| normal | 1 | 143 | 3 | 1304 | 187 | 97 |
| other | 39 | 139 | 288 | 14 | 1100 | 137 |
| stroma | 3 | 77 | 0 | 87 | 296 | 619 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7773    0.8593    0.8162       199
      cancer     0.8702    0.9229    0.8958      2645
       empty     0.9168    0.9954    0.9545      3256
      normal     0.8987    0.7516    0.8186      1735
       other     0.6407    0.6407    0.6407      1717
      stroma     0.6832    0.5721    0.6227      1082

    accuracy                         0.8347     10634
   macro avg     0.7978    0.7903    0.7914     10634
weighted avg     0.8313    0.8347    0.8307     10634

```