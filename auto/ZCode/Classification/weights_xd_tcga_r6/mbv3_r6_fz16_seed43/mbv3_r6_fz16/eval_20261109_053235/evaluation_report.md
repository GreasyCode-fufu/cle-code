# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_r6_fz16_seed43/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7999 |
| Accuracy | 0.8499 |
| Macro-AUROC (OvR) | 0.9702 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8476 |
| Balanced Acc | 0.8015 |
| Macro-Sensitivity | 0.8015 |
| Macro-Specificity | 0.9693 |
| Macro-Precision | 0.8054 |
| AUC-PR (Macro) | 0.8564 |
| Cohen's Kappa | 0.8094 |
| MCC | 0.8113 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.92M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 7195.4 (0.14 ms/张) |
| End-to-End FPS | 55.4 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7500 | 0.9930 | 0.6579 | 0.7009 | 0.9955 | 0.9830 | 0.7744 | 100 |
| cancer | 0.9202 | 0.9104 | 0.7573 | 0.8308 | 0.9741 | 0.9723 | 0.9207 | 1329 |
| empty | 0.9944 | 0.9961 | 0.9902 | 0.9923 | 0.9978 | 0.9999 | 0.9997 | 1621 |
| normal | 0.8797 | 0.9870 | 0.9360 | 0.9070 | 0.9743 | 0.9926 | 0.9707 | 1014 |
| other | 0.6997 | 0.9770 | 0.8471 | 0.7664 | 0.9469 | 0.9493 | 0.8597 | 879 |
| stroma | 0.5652 | 0.9521 | 0.6441 | 0.6021 | 0.9345 | 0.9238 | 0.6136 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=75 FP=39 TN=5563 FN=25
- **cancer**: TP=1223 FP=392 TN=3981 FN=106
- **empty**: TP=1612 FP=16 TN=4065 FN=9
- **normal**: TP=892 FP=61 TN=4627 FN=122
- **other**: TP=615 FP=111 TN=4712 FN=264
- **stroma**: TP=429 FP=237 TN=4706 FN=330

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 75 | 3 | 2 | 5 | 7 | 8 |
| cancer | 20 | 1223 | 1 | 20 | 22 | 43 |
| empty | 0 | 1 | 1612 | 0 | 6 | 2 |
| normal | 2 | 60 | 0 | 892 | 12 | 48 |
| other | 7 | 94 | 12 | 15 | 615 | 136 |
| stroma | 10 | 234 | 1 | 21 | 64 | 429 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6579    0.7500    0.7009       100
      cancer     0.7573    0.9202    0.8308      1329
       empty     0.9902    0.9944    0.9923      1621
      normal     0.9360    0.8797    0.9070      1014
       other     0.8471    0.6997    0.7664       879
      stroma     0.6441    0.5652    0.6021       759

    accuracy                         0.8499      5702
   macro avg     0.8054    0.8015    0.7999      5702
weighted avg     0.8523    0.8499    0.8476      5702

```