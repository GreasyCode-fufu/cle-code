# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_dce4m_seed44/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7922 |
| Accuracy | 0.8369 |
| Macro-AUROC (OvR) | 0.9549 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8325 |
| Balanced Acc | 0.7996 |
| Macro-Sensitivity | 0.7996 |
| Macro-Specificity | 0.9662 |
| Macro-Precision | 0.7911 |
| AUC-PR (Macro) | 0.8577 |
| Cohen's Kappa | 0.7898 |
| MCC | 0.7907 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9253.7 (0.11 ms/张) |
| End-to-End FPS | 554.0 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8894 | 0.9935 | 0.7224 | 0.7973 | 0.9979 | 0.9969 | 0.9153 | 199 |
| cancer | 0.9202 | 0.9527 | 0.8656 | 0.8921 | 0.9730 | 0.9885 | 0.9680 | 2645 |
| empty | 0.9966 | 0.9603 | 0.9172 | 0.9553 | 0.9984 | 0.9816 | 0.9310 | 3256 |
| normal | 0.7539 | 0.9798 | 0.8790 | 0.8117 | 0.9533 | 0.9425 | 0.8707 | 1735 |
| other | 0.6249 | 0.9430 | 0.6787 | 0.6507 | 0.9289 | 0.8752 | 0.7260 | 1717 |
| stroma | 0.6128 | 0.9679 | 0.6835 | 0.6462 | 0.9566 | 0.9448 | 0.7350 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=177 FP=68 TN=10367 FN=22
- **cancer**: TP=2434 FP=378 TN=7611 FN=211
- **empty**: TP=3245 FP=293 TN=7085 FN=11
- **normal**: TP=1308 FP=180 TN=8719 FN=427
- **other**: TP=1073 FP=508 TN=8409 FN=644
- **stroma**: TP=663 FP=307 TN=9245 FN=419

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 177 | 4 | 2 | 3 | 11 | 2 |
| cancer | 7 | 2434 | 0 | 80 | 76 | 48 |
| empty | 0 | 2 | 3245 | 0 | 9 | 0 |
| normal | 2 | 151 | 3 | 1308 | 166 | 105 |
| other | 54 | 135 | 288 | 15 | 1073 | 152 |
| stroma | 5 | 86 | 0 | 82 | 246 | 663 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7224    0.8894    0.7973       199
      cancer     0.8656    0.9202    0.8921      2645
       empty     0.9172    0.9966    0.9553      3256
      normal     0.8790    0.7539    0.8117      1735
       other     0.6787    0.6249    0.6507      1717
      stroma     0.6835    0.6128    0.6462      1082

    accuracy                         0.8369     10634
   macro avg     0.7911    0.7996    0.7922     10634
weighted avg     0.8322    0.8369    0.8325     10634

```