# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_tiny_seed42/convnext_tiny/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8381 |
| Accuracy | 0.8737 |
| Macro-AUROC (OvR) | 0.9777 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8758 |
| Balanced Acc | 0.8509 |
| Macro-Sensitivity | 0.8509 |
| Macro-Specificity | 0.9752 |
| Macro-Precision | 0.8337 |
| AUC-PR (Macro) | 0.8965 |
| Cohen's Kappa | 0.8414 |
| MCC | 0.8435 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 614.9 (1.63 ms/张) |
| End-to-End FPS | 320.1 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7800 | 0.9943 | 0.7091 | 0.7429 | 0.9961 | 0.9962 | 0.8641 | 100 |
| cancer | 0.7457 | 0.9861 | 0.9420 | 0.8324 | 0.9273 | 0.9640 | 0.9222 | 1329 |
| empty | 0.9963 | 0.9973 | 0.9932 | 0.9948 | 0.9985 | 0.9998 | 0.9996 | 1621 |
| normal | 0.9497 | 0.9895 | 0.9516 | 0.9506 | 0.9891 | 0.9928 | 0.9823 | 1014 |
| other | 0.7918 | 0.9525 | 0.7524 | 0.7716 | 0.9617 | 0.9538 | 0.8637 | 879 |
| stroma | 0.8419 | 0.9316 | 0.6540 | 0.7362 | 0.9746 | 0.9598 | 0.7469 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=78 FP=32 TN=5570 FN=22
- **cancer**: TP=991 FP=61 TN=4312 FN=338
- **empty**: TP=1615 FP=11 TN=4070 FN=6
- **normal**: TP=963 FP=49 TN=4639 FN=51
- **other**: TP=696 FP=229 TN=4594 FN=183
- **stroma**: TP=639 FP=338 TN=4605 FN=120

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 78 | 3 | 1 | 5 | 9 | 4 |
| cancer | 28 | 991 | 1 | 15 | 124 | 170 |
| empty | 0 | 0 | 1615 | 2 | 4 | 0 |
| normal | 2 | 9 | 0 | 963 | 6 | 34 |
| other | 1 | 34 | 9 | 9 | 696 | 130 |
| stroma | 1 | 15 | 0 | 18 | 86 | 639 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7091    0.7800    0.7429       100
      cancer     0.9420    0.7457    0.8324      1329
       empty     0.9932    0.9963    0.9948      1621
      normal     0.9516    0.9497    0.9506      1014
       other     0.7524    0.7918    0.7716       879
      stroma     0.6540    0.8419    0.7362       759

    accuracy                         0.8737      5702
   macro avg     0.8337    0.8509    0.8381      5702
weighted avg     0.8866    0.8737    0.8758      5702

```