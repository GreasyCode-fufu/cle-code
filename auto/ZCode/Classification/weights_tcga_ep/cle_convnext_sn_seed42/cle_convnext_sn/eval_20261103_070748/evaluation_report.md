# cle_convnext_sn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/cle_convnext_sn_seed42/cle_convnext_sn/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8096 |
| Accuracy | 0.8440 |
| Macro-AUROC (OvR) | 0.9562 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8373 |
| Balanced Acc | 0.8092 |
| Macro-Sensitivity | 0.8092 |
| Macro-Specificity | 0.9670 |
| Macro-Precision | 0.8208 |
| AUC-PR (Macro) | 0.8642 |
| Cohen's Kappa | 0.7984 |
| MCC | 0.8011 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 603.6 (1.66 ms/张) |
| End-to-End FPS | 315.8 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8794 | 0.9960 | 0.8065 | 0.8413 | 0.9977 | 0.9976 | 0.9230 | 199 |
| cancer | 0.9410 | 0.9229 | 0.8016 | 0.8657 | 0.9793 | 0.9765 | 0.9252 | 2645 |
| empty | 0.9963 | 0.9619 | 0.9203 | 0.9568 | 0.9983 | 0.9773 | 0.9179 | 3256 |
| normal | 0.7810 | 0.9829 | 0.8991 | 0.8359 | 0.9584 | 0.9637 | 0.9010 | 1735 |
| other | 0.5539 | 0.9702 | 0.7814 | 0.6483 | 0.9187 | 0.8727 | 0.7291 | 1717 |
| stroma | 0.7033 | 0.9684 | 0.7159 | 0.7096 | 0.9665 | 0.9493 | 0.7888 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=175 FP=42 TN=10393 FN=24
- **cancer**: TP=2489 FP=616 TN=7373 FN=156
- **empty**: TP=3244 FP=281 TN=7097 FN=12
- **normal**: TP=1355 FP=152 TN=8747 FN=380
- **other**: TP=951 FP=266 TN=8651 FN=766
- **stroma**: TP=761 FP=302 TN=9250 FN=321

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 175 | 9 | 2 | 6 | 6 | 1 |
| cancer | 3 | 2489 | 2 | 44 | 35 | 72 |
| empty | 0 | 0 | 3244 | 0 | 12 | 0 |
| normal | 1 | 194 | 0 | 1355 | 91 | 94 |
| other | 36 | 295 | 276 | 24 | 951 | 135 |
| stroma | 2 | 118 | 1 | 78 | 122 | 761 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8065    0.8794    0.8413       199
      cancer     0.8016    0.9410    0.8657      2645
       empty     0.9203    0.9963    0.9568      3256
      normal     0.8991    0.7810    0.8359      1735
       other     0.7814    0.5539    0.6483      1717
      stroma     0.7159    0.7033    0.7096      1082

    accuracy                         0.8440     10634
   macro avg     0.8208    0.8092    0.8096     10634
weighted avg     0.8420    0.8440    0.8373     10634

```