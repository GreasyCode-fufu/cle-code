# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce4m_seed43/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7848 |
| Accuracy | 0.8273 |
| Macro-AUROC (OvR) | 0.9422 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8225 |
| Balanced Acc | 0.7874 |
| Macro-Sensitivity | 0.7874 |
| Macro-Specificity | 0.9641 |
| Macro-Precision | 0.7884 |
| AUC-PR (Macro) | 0.8379 |
| Cohen's Kappa | 0.7773 |
| MCC | 0.7787 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9207.6 (0.11 ms/张) |
| End-to-End FPS | 552.5 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8442 | 0.9947 | 0.7534 | 0.7962 | 0.9970 | 0.9955 | 0.9074 | 199 |
| cancer | 0.9108 | 0.9377 | 0.8287 | 0.8678 | 0.9695 | 0.9777 | 0.9296 | 2645 |
| empty | 0.9957 | 0.9610 | 0.9184 | 0.9555 | 0.9980 | 0.9822 | 0.9371 | 3256 |
| normal | 0.7182 | 0.9817 | 0.8843 | 0.7926 | 0.9470 | 0.9180 | 0.8525 | 1735 |
| other | 0.5882 | 0.9491 | 0.6899 | 0.6350 | 0.9229 | 0.8437 | 0.6915 | 1717 |
| stroma | 0.6673 | 0.9603 | 0.6558 | 0.6615 | 0.9622 | 0.9364 | 0.7095 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=168 FP=55 TN=10380 FN=31
- **cancer**: TP=2409 FP=498 TN=7491 FN=236
- **empty**: TP=3242 FP=288 TN=7090 FN=14
- **normal**: TP=1246 FP=163 TN=8736 FN=489
- **other**: TP=1010 FP=454 TN=8463 FN=707
- **stroma**: TP=722 FP=379 TN=9173 FN=360

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 168 | 5 | 2 | 9 | 12 | 3 |
| cancer | 9 | 2409 | 1 | 48 | 100 | 78 |
| empty | 0 | 0 | 3242 | 0 | 8 | 6 |
| normal | 0 | 192 | 1 | 1246 | 160 | 136 |
| other | 42 | 202 | 284 | 23 | 1010 | 156 |
| stroma | 4 | 99 | 0 | 83 | 174 | 722 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7534    0.8442    0.7962       199
      cancer     0.8287    0.9108    0.8678      2645
       empty     0.9184    0.9957    0.9555      3256
      normal     0.8843    0.7182    0.7926      1735
       other     0.6899    0.5882    0.6350      1717
      stroma     0.6558    0.6673    0.6615      1082

    accuracy                         0.8273     10634
   macro avg     0.7884    0.7874    0.7848     10634
weighted avg     0.8238    0.8273    0.8225     10634

```