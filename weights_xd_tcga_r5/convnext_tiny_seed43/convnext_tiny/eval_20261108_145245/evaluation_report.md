# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r5/convnext_tiny_seed43/convnext_tiny/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8077 |
| Accuracy | 0.8471 |
| Macro-AUROC (OvR) | 0.9592 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8431 |
| Balanced Acc | 0.8081 |
| Macro-Sensitivity | 0.8081 |
| Macro-Specificity | 0.9682 |
| Macro-Precision | 0.8143 |
| AUC-PR (Macro) | 0.8605 |
| Cohen's Kappa | 0.8028 |
| MCC | 0.8042 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1826.6 (0.55 ms/张) |
| End-to-End FPS | 490.2 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8643 | 0.9951 | 0.7713 | 0.8152 | 0.9974 | 0.9968 | 0.8936 | 199 |
| cancer | 0.9512 | 0.9567 | 0.8791 | 0.9137 | 0.9834 | 0.9875 | 0.9726 | 2645 |
| empty | 0.9982 | 0.9589 | 0.9147 | 0.9546 | 0.9992 | 0.9844 | 0.9625 | 3256 |
| normal | 0.7170 | 0.9899 | 0.9325 | 0.8107 | 0.9472 | 0.9383 | 0.8701 | 1735 |
| other | 0.6302 | 0.9373 | 0.6594 | 0.6444 | 0.9294 | 0.9044 | 0.7081 | 1717 |
| stroma | 0.6876 | 0.9710 | 0.7287 | 0.7076 | 0.9648 | 0.9438 | 0.7561 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=172 FP=51 TN=10384 FN=27
- **cancer**: TP=2516 FP=346 TN=7643 FN=129
- **empty**: TP=3250 FP=303 TN=7075 FN=6
- **normal**: TP=1244 FP=90 TN=8809 FN=491
- **other**: TP=1082 FP=559 TN=8358 FN=635
- **stroma**: TP=744 FP=277 TN=9275 FN=338

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 172 | 6 | 4 | 4 | 12 | 1 |
| cancer | 7 | 2516 | 3 | 11 | 34 | 74 |
| empty | 0 | 2 | 3250 | 0 | 4 | 0 |
| normal | 2 | 88 | 2 | 1244 | 301 | 98 |
| other | 40 | 176 | 294 | 21 | 1082 | 104 |
| stroma | 2 | 74 | 0 | 54 | 208 | 744 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7713    0.8643    0.8152       199
      cancer     0.8791    0.9512    0.9137      2645
       empty     0.9147    0.9982    0.9546      3256
      normal     0.9325    0.7170    0.8107      1735
       other     0.6594    0.6302    0.6444      1717
      stroma     0.7287    0.6876    0.7076      1082

    accuracy                         0.8471     10634
   macro avg     0.8143    0.8081    0.8077     10634
weighted avg     0.8459    0.8471    0.8431     10634

```