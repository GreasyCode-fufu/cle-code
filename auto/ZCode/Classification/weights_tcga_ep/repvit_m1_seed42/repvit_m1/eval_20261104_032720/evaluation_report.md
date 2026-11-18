# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/repvit_m1_seed42/repvit_m1/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8088 |
| Accuracy | 0.8409 |
| Macro-AUROC (OvR) | 0.9658 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8442 |
| Balanced Acc | 0.8351 |
| Macro-Sensitivity | 0.8351 |
| Macro-Specificity | 0.9685 |
| Macro-Precision | 0.7954 |
| AUC-PR (Macro) | 0.8705 |
| Cohen's Kappa | 0.8000 |
| MCC | 0.8013 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 2502.1 (0.40 ms/张) |
| End-to-End FPS | 513.4 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.9200 | 0.9913 | 0.6525 | 0.7635 | 0.9986 | 0.9962 | 0.8761 | 100 |
| cancer | 0.7938 | 0.9751 | 0.9064 | 0.8464 | 0.9396 | 0.9683 | 0.9224 | 1329 |
| empty | 0.9926 | 0.9936 | 0.9841 | 0.9883 | 0.9970 | 0.9996 | 0.9990 | 1621 |
| normal | 0.7899 | 0.9844 | 0.9165 | 0.8485 | 0.9559 | 0.9877 | 0.9475 | 1014 |
| other | 0.7406 | 0.9262 | 0.6465 | 0.6903 | 0.9514 | 0.9055 | 0.7693 | 879 |
| stroma | 0.7734 | 0.9405 | 0.6663 | 0.7159 | 0.9643 | 0.9375 | 0.7087 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=92 FP=49 TN=5553 FN=8
- **cancer**: TP=1055 FP=109 TN=4264 FN=274
- **empty**: TP=1609 FP=26 TN=4055 FN=12
- **normal**: TP=801 FP=73 TN=4615 FN=213
- **other**: TP=651 FP=356 TN=4467 FN=228
- **stroma**: TP=587 FP=294 TN=4649 FN=172

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 92 | 1 | 0 | 1 | 3 | 3 |
| cancer | 14 | 1055 | 0 | 47 | 120 | 93 |
| empty | 1 | 1 | 1609 | 0 | 8 | 2 |
| normal | 7 | 12 | 0 | 801 | 131 | 63 |
| other | 12 | 52 | 25 | 6 | 651 | 133 |
| stroma | 15 | 43 | 1 | 19 | 94 | 587 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6525    0.9200    0.7635       100
      cancer     0.9064    0.7938    0.8464      1329
       empty     0.9841    0.9926    0.9883      1621
      normal     0.9165    0.7899    0.8485      1014
       other     0.6465    0.7406    0.6903       879
      stroma     0.6663    0.7734    0.7159       759

    accuracy                         0.8409      5702
   macro avg     0.7954    0.8351    0.8088      5702
weighted avg     0.8538    0.8409    0.8442      5702

```