# convnext_egb_scg_hsc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_egb_scg_hsc_seed43/convnext_egb_scg_hsc/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8533 |
| Accuracy | 0.8871 |
| Macro-AUROC (OvR) | 0.9794 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8892 |
| Balanced Acc | 0.8611 |
| Macro-Sensitivity | 0.8611 |
| Macro-Specificity | 0.9777 |
| Macro-Precision | 0.8512 |
| AUC-PR (Macro) | 0.9066 |
| Cohen's Kappa | 0.8578 |
| MCC | 0.8591 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.58M |
| Pure-Inference FPS | 503.0 (1.99 ms/张) |
| End-to-End FPS | 286.8 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7900 | 0.9952 | 0.7453 | 0.7670 | 0.9962 | 0.9959 | 0.8732 | 100 |
| cancer | 0.8172 | 0.9803 | 0.9266 | 0.8685 | 0.9464 | 0.9663 | 0.9319 | 1329 |
| empty | 0.9951 | 0.9978 | 0.9945 | 0.9948 | 0.9980 | 0.9998 | 0.9995 | 1621 |
| normal | 0.9428 | 0.9872 | 0.9409 | 0.9419 | 0.9876 | 0.9957 | 0.9825 | 1014 |
| other | 0.7759 | 0.9737 | 0.8430 | 0.8081 | 0.9597 | 0.9557 | 0.8870 | 879 |
| stroma | 0.8458 | 0.9322 | 0.6571 | 0.7396 | 0.9752 | 0.9628 | 0.7658 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=79 FP=27 TN=5575 FN=21
- **cancer**: TP=1086 FP=86 TN=4287 FN=243
- **empty**: TP=1613 FP=9 TN=4072 FN=8
- **normal**: TP=956 FP=60 TN=4628 FN=58
- **other**: TP=682 FP=127 TN=4696 FN=197
- **stroma**: TP=642 FP=335 TN=4608 FN=117

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 79 | 0 | 1 | 5 | 10 | 5 |
| cancer | 22 | 1086 | 1 | 19 | 32 | 169 |
| empty | 0 | 0 | 1613 | 0 | 6 | 2 |
| normal | 2 | 19 | 0 | 956 | 9 | 28 |
| other | 2 | 43 | 7 | 14 | 682 | 131 |
| stroma | 1 | 24 | 0 | 22 | 70 | 642 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7453    0.7900    0.7670       100
      cancer     0.9266    0.8172    0.8685      1329
       empty     0.9945    0.9951    0.9948      1621
      normal     0.9409    0.9428    0.9419      1014
       other     0.8430    0.7759    0.8081       879
      stroma     0.6571    0.8458    0.7396       759

    accuracy                         0.8871      5702
   macro avg     0.8512    0.8611    0.8533      5702
weighted avg     0.8965    0.8871    0.8892      5702

```