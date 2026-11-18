# convnext_tiny_cle_oca_strip 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_tiny_cle_oca_strip_seed42/convnext_tiny_cle_oca_strip/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8165 |
| Accuracy | 0.8475 |
| Macro-AUROC (OvR) | 0.9713 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8457 |
| Balanced Acc | 0.8230 |
| Macro-Sensitivity | 0.8230 |
| Macro-Specificity | 0.9686 |
| Macro-Precision | 0.8143 |
| AUC-PR (Macro) | 0.8902 |
| Cohen's Kappa | 0.8043 |
| MCC | 0.8051 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.47G |
| FLOPs | 8.94G |
| Params | 27.84M |
| Pure-Inference FPS | 529.3 (1.89 ms/张) |
| End-to-End FPS | 293.7 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8794 | 0.9959 | 0.8028 | 0.8393 | 0.9977 | 0.9961 | 0.9266 | 199 |
| cancer | 0.8752 | 0.9643 | 0.8904 | 0.8827 | 0.9589 | 0.9775 | 0.9503 | 2645 |
| empty | 0.9923 | 0.9626 | 0.9213 | 0.9555 | 0.9965 | 0.9860 | 0.9685 | 3256 |
| normal | 0.7896 | 0.9813 | 0.8919 | 0.8377 | 0.9599 | 0.9734 | 0.9141 | 1735 |
| other | 0.6377 | 0.9529 | 0.7228 | 0.6776 | 0.9318 | 0.9294 | 0.7805 | 1717 |
| stroma | 0.7634 | 0.9548 | 0.6566 | 0.7060 | 0.9727 | 0.9652 | 0.8015 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=175 FP=43 TN=10392 FN=24
- **cancer**: TP=2315 FP=285 TN=7704 FN=330
- **empty**: TP=3231 FP=276 TN=7102 FN=25
- **normal**: TP=1370 FP=166 TN=8733 FN=365
- **other**: TP=1095 FP=420 TN=8497 FN=622
- **stroma**: TP=826 FP=432 TN=9120 FN=256

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 175 | 8 | 3 | 5 | 7 | 1 |
| cancer | 4 | 2315 | 0 | 68 | 126 | 132 |
| empty | 0 | 1 | 3231 | 0 | 24 | 0 |
| normal | 0 | 122 | 0 | 1370 | 116 | 127 |
| other | 36 | 106 | 273 | 35 | 1095 | 172 |
| stroma | 3 | 48 | 0 | 58 | 147 | 826 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8028    0.8794    0.8393       199
      cancer     0.8904    0.8752    0.8827      2645
       empty     0.9213    0.9923    0.9555      3256
      normal     0.8919    0.7896    0.8377      1735
       other     0.7228    0.6377    0.6776      1717
      stroma     0.6566    0.7634    0.7060      1082

    accuracy                         0.8475     10634
   macro avg     0.8143    0.8230    0.8165     10634
weighted avg     0.8476    0.8475    0.8457     10634

```