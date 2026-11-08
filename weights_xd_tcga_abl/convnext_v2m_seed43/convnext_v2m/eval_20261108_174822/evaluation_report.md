# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_abl/convnext_v2m_seed43/convnext_v2m/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8295 |
| Accuracy | 0.8590 |
| Macro-AUROC (OvR) | 0.9750 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8591 |
| Balanced Acc | 0.8281 |
| Macro-Sensitivity | 0.8281 |
| Macro-Specificity | 0.9714 |
| Macro-Precision | 0.8319 |
| AUC-PR (Macro) | 0.8905 |
| Cohen's Kappa | 0.8216 |
| MCC | 0.8219 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 603.3 (1.66 ms/张) |
| End-to-End FPS | 315.4 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8100 | 0.9964 | 0.8020 | 0.8060 | 0.9966 | 0.9951 | 0.8800 | 100 |
| cancer | 0.8465 | 0.9321 | 0.7911 | 0.8179 | 0.9523 | 0.9634 | 0.9126 | 1329 |
| empty | 0.9907 | 0.9993 | 0.9981 | 0.9944 | 0.9963 | 1.0000 | 0.9999 | 1621 |
| normal | 0.9152 | 0.9915 | 0.9587 | 0.9364 | 0.9818 | 0.9952 | 0.9824 | 1014 |
| other | 0.7565 | 0.9509 | 0.7373 | 0.7468 | 0.9554 | 0.9511 | 0.8540 | 879 |
| stroma | 0.6495 | 0.9581 | 0.7043 | 0.6758 | 0.9468 | 0.9453 | 0.7140 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=81 FP=20 TN=5582 FN=19
- **cancer**: TP=1125 FP=297 TN=4076 FN=204
- **empty**: TP=1606 FP=3 TN=4078 FN=15
- **normal**: TP=928 FP=40 TN=4648 FN=86
- **other**: TP=665 FP=237 TN=4586 FN=214
- **stroma**: TP=493 FP=207 TN=4736 FN=266

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 81 | 1 | 0 | 4 | 12 | 2 |
| cancer | 17 | 1125 | 0 | 12 | 109 | 66 |
| empty | 0 | 0 | 1606 | 1 | 14 | 0 |
| normal | 2 | 36 | 0 | 928 | 2 | 46 |
| other | 0 | 110 | 3 | 8 | 665 | 93 |
| stroma | 1 | 150 | 0 | 15 | 100 | 493 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8020    0.8100    0.8060       100
      cancer     0.7911    0.8465    0.8179      1329
       empty     0.9981    0.9907    0.9944      1621
      normal     0.9587    0.9152    0.9364      1014
       other     0.7373    0.7565    0.7468       879
      stroma     0.7043    0.6495    0.6758       759

    accuracy                         0.8590      5702
   macro avg     0.8319    0.8281    0.8295      5702
weighted avg     0.8601    0.8590    0.8591      5702

```