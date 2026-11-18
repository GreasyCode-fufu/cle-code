# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/resnet50_seed42/resnet50/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8095 |
| Accuracy | 0.8446 |
| Macro-AUROC (OvR) | 0.9623 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8416 |
| Balanced Acc | 0.8113 |
| Macro-Sensitivity | 0.8113 |
| Macro-Specificity | 0.9678 |
| Macro-Precision | 0.8145 |
| AUC-PR (Macro) | 0.8689 |
| Cohen's Kappa | 0.7999 |
| MCC | 0.8005 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.52M |
| Pure-Inference FPS | 945.5 (1.06 ms/张) |
| End-to-End FPS | 386.3 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8995 | 0.9949 | 0.7716 | 0.8306 | 0.9981 | 0.9971 | 0.9100 | 199 |
| cancer | 0.9172 | 0.9667 | 0.9012 | 0.9091 | 0.9724 | 0.9887 | 0.9703 | 2645 |
| empty | 0.9899 | 0.9620 | 0.9201 | 0.9537 | 0.9954 | 0.9924 | 0.9789 | 3256 |
| normal | 0.7764 | 0.9649 | 0.8119 | 0.7938 | 0.9568 | 0.9598 | 0.8630 | 1735 |
| other | 0.6564 | 0.9325 | 0.6518 | 0.6541 | 0.9337 | 0.8928 | 0.6978 | 1717 |
| stroma | 0.6285 | 0.9854 | 0.8303 | 0.7154 | 0.9590 | 0.9432 | 0.7933 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=179 FP=53 TN=10382 FN=20
- **cancer**: TP=2426 FP=266 TN=7723 FN=219
- **empty**: TP=3223 FP=280 TN=7098 FN=33
- **normal**: TP=1347 FP=312 TN=8587 FN=388
- **other**: TP=1127 FP=602 TN=8315 FN=590
- **stroma**: TP=680 FP=139 TN=9413 FN=402

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 179 | 3 | 9 | 3 | 5 | 0 |
| cancer | 3 | 2426 | 0 | 106 | 86 | 24 |
| empty | 0 | 6 | 3223 | 0 | 27 | 0 |
| normal | 6 | 77 | 0 | 1347 | 269 | 36 |
| other | 42 | 117 | 271 | 81 | 1127 | 79 |
| stroma | 2 | 63 | 0 | 122 | 215 | 680 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7716    0.8995    0.8306       199
      cancer     0.9012    0.9172    0.9091      2645
       empty     0.9201    0.9899    0.9537      3256
      normal     0.8119    0.7764    0.7938      1735
       other     0.6518    0.6564    0.6541      1717
      stroma     0.8303    0.6285    0.7154      1082

    accuracy                         0.8446     10634
   macro avg     0.8145    0.8113    0.8095     10634
weighted avg     0.8425    0.8446    0.8416     10634

```