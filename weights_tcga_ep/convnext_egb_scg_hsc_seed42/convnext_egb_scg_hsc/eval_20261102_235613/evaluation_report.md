# convnext_egb_scg_hsc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_egb_scg_hsc_seed42/convnext_egb_scg_hsc/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8783 |
| Accuracy | 0.9071 |
| Macro-AUROC (OvR) | 0.9871 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9081 |
| Balanced Acc | 0.8843 |
| Macro-Sensitivity | 0.8843 |
| Macro-Specificity | 0.9817 |
| Macro-Precision | 0.8744 |
| AUC-PR (Macro) | 0.9318 |
| Cohen's Kappa | 0.8828 |
| MCC | 0.8832 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.58M |
| Pure-Inference FPS | 343.2 (2.91 ms/张) |
| End-to-End FPS | 226.8 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8400 | 0.9963 | 0.8000 | 0.8195 | 0.9971 | 0.9981 | 0.9136 | 100 |
| cancer | 0.8849 | 0.9847 | 0.9461 | 0.9145 | 0.9657 | 0.9860 | 0.9683 | 1329 |
| empty | 0.9914 | 0.9993 | 0.9981 | 0.9947 | 0.9966 | 0.9999 | 0.9998 | 1621 |
| normal | 0.9576 | 0.9866 | 0.9391 | 0.9482 | 0.9908 | 0.9950 | 0.9843 | 1014 |
| other | 0.7952 | 0.9720 | 0.8381 | 0.8161 | 0.9630 | 0.9713 | 0.9060 | 879 |
| stroma | 0.8366 | 0.9512 | 0.7249 | 0.7768 | 0.9743 | 0.9722 | 0.8187 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=84 FP=21 TN=5581 FN=16
- **cancer**: TP=1176 FP=67 TN=4306 FN=153
- **empty**: TP=1607 FP=3 TN=4078 FN=14
- **normal**: TP=971 FP=63 TN=4625 FN=43
- **other**: TP=699 FP=135 TN=4688 FN=180
- **stroma**: TP=635 FP=241 TN=4702 FN=124

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 84 | 1 | 0 | 2 | 9 | 4 |
| cancer | 18 | 1176 | 0 | 28 | 23 | 84 |
| empty | 0 | 0 | 1607 | 0 | 12 | 2 |
| normal | 2 | 10 | 0 | 971 | 0 | 31 |
| other | 0 | 42 | 3 | 15 | 699 | 120 |
| stroma | 1 | 14 | 0 | 18 | 91 | 635 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8000    0.8400    0.8195       100
      cancer     0.9461    0.8849    0.9145      1329
       empty     0.9981    0.9914    0.9947      1621
      normal     0.9391    0.9576    0.9482      1014
       other     0.8381    0.7952    0.8161       879
      stroma     0.7249    0.8366    0.7768       759

    accuracy                         0.9071      5702
   macro avg     0.8744    0.8843    0.8783      5702
weighted avg     0.9110    0.9071    0.9081      5702

```