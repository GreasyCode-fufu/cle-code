# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_small_seed43/convnext_small/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8183 |
| Accuracy | 0.8599 |
| Macro-AUROC (OvR) | 0.9738 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8598 |
| Balanced Acc | 0.8345 |
| Macro-Sensitivity | 0.8345 |
| Macro-Specificity | 0.9721 |
| Macro-Precision | 0.8085 |
| AUC-PR (Macro) | 0.8798 |
| Cohen's Kappa | 0.8235 |
| MCC | 0.8246 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.42M |
| Pure-Inference FPS | 358.1 (2.79 ms/张) |
| End-to-End FPS | 232.6 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8000 | 0.9921 | 0.6452 | 0.7143 | 0.9964 | 0.9951 | 0.8483 | 100 |
| cancer | 0.7524 | 0.9689 | 0.8803 | 0.8114 | 0.9279 | 0.9543 | 0.8965 | 1329 |
| empty | 0.9957 | 0.9973 | 0.9932 | 0.9945 | 0.9983 | 0.9999 | 0.9998 | 1621 |
| normal | 0.9744 | 0.9785 | 0.9073 | 0.9396 | 0.9944 | 0.9959 | 0.9834 | 1014 |
| other | 0.7861 | 0.9322 | 0.6788 | 0.7285 | 0.9599 | 0.9458 | 0.7995 | 879 |
| stroma | 0.6983 | 0.9636 | 0.7465 | 0.7216 | 0.9541 | 0.9516 | 0.7515 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=80 FP=44 TN=5558 FN=20
- **cancer**: TP=1000 FP=136 TN=4237 FN=329
- **empty**: TP=1614 FP=11 TN=4070 FN=7
- **normal**: TP=988 FP=101 TN=4587 FN=26
- **other**: TP=691 FP=327 TN=4496 FN=188
- **stroma**: TP=530 FP=180 TN=4763 FN=229

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 80 | 2 | 2 | 8 | 5 | 3 |
| cancer | 39 | 1000 | 1 | 56 | 170 | 63 |
| empty | 0 | 0 | 1614 | 0 | 5 | 2 |
| normal | 1 | 12 | 0 | 988 | 0 | 13 |
| other | 2 | 66 | 8 | 13 | 691 | 99 |
| stroma | 2 | 56 | 0 | 24 | 147 | 530 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6452    0.8000    0.7143       100
      cancer     0.8803    0.7524    0.8114      1329
       empty     0.9932    0.9957    0.9945      1621
      normal     0.9073    0.9744    0.9396      1014
       other     0.6788    0.7861    0.7285       879
      stroma     0.7465    0.6983    0.7216       759

    accuracy                         0.8599      5702
   macro avg     0.8085    0.8345    0.8183      5702
weighted avg     0.8642    0.8599    0.8598      5702

```