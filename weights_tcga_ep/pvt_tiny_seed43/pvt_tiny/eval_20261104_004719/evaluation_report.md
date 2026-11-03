# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/pvt_tiny_seed43/pvt_tiny/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7948 |
| Accuracy | 0.8518 |
| Macro-AUROC (OvR) | 0.9729 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8527 |
| Balanced Acc | 0.8189 |
| Macro-Sensitivity | 0.8189 |
| Macro-Specificity | 0.9703 |
| Macro-Precision | 0.7847 |
| AUC-PR (Macro) | 0.8699 |
| Cohen's Kappa | 0.8129 |
| MCC | 0.8136 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 1282.3 (0.78 ms/张) |
| End-to-End FPS | 432.0 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8000 | 0.9861 | 0.5063 | 0.6202 | 0.9964 | 0.9867 | 0.7612 | 100 |
| cancer | 0.8781 | 0.9408 | 0.8184 | 0.8472 | 0.9621 | 0.9708 | 0.9302 | 1329 |
| empty | 0.9957 | 0.9968 | 0.9920 | 0.9938 | 0.9983 | 0.9997 | 0.9994 | 1621 |
| normal | 0.7880 | 0.9893 | 0.9411 | 0.8578 | 0.9557 | 0.9893 | 0.9544 | 1014 |
| other | 0.7952 | 0.9533 | 0.7565 | 0.7754 | 0.9623 | 0.9477 | 0.8642 | 879 |
| stroma | 0.6561 | 0.9555 | 0.6936 | 0.6743 | 0.9476 | 0.9432 | 0.7101 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=80 FP=78 TN=5524 FN=20
- **cancer**: TP=1167 FP=259 TN=4114 FN=162
- **empty**: TP=1614 FP=13 TN=4068 FN=7
- **normal**: TP=799 FP=50 TN=4638 FN=215
- **other**: TP=699 FP=225 TN=4598 FN=180
- **stroma**: TP=498 FP=220 TN=4723 FN=261

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 80 | 2 | 3 | 2 | 9 | 4 |
| cancer | 7 | 1167 | 0 | 26 | 38 | 91 |
| empty | 0 | 0 | 1614 | 0 | 7 | 0 |
| normal | 3 | 117 | 0 | 799 | 69 | 26 |
| other | 14 | 52 | 9 | 6 | 699 | 99 |
| stroma | 54 | 88 | 1 | 16 | 102 | 498 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.5063    0.8000    0.6202       100
      cancer     0.8184    0.8781    0.8472      1329
       empty     0.9920    0.9957    0.9938      1621
      normal     0.9411    0.7880    0.8578      1014
       other     0.7565    0.7952    0.7754       879
      stroma     0.6936    0.6561    0.6743       759

    accuracy                         0.8518      5702
   macro avg     0.7847    0.8189    0.7948      5702
weighted avg     0.8579    0.8518    0.8527      5702

```