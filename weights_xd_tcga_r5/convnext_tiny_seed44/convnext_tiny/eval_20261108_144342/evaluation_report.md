# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r5/convnext_tiny_seed44/convnext_tiny/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8461 |
| Accuracy | 0.8862 |
| Macro-AUROC (OvR) | 0.9815 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8864 |
| Balanced Acc | 0.8553 |
| Macro-Sensitivity | 0.8553 |
| Macro-Specificity | 0.9774 |
| Macro-Precision | 0.8385 |
| AUC-PR (Macro) | 0.9067 |
| Cohen's Kappa | 0.8565 |
| MCC | 0.8567 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 919.0 (1.09 ms/张) |
| End-to-End FPS | 385.4 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7900 | 0.9939 | 0.6991 | 0.7418 | 0.9962 | 0.9956 | 0.8645 | 100 |
| cancer | 0.8435 | 0.9744 | 0.9092 | 0.8751 | 0.9535 | 0.9772 | 0.9400 | 1329 |
| empty | 0.9926 | 0.9980 | 0.9951 | 0.9938 | 0.9971 | 0.9999 | 0.9997 | 1621 |
| normal | 0.9734 | 0.9831 | 0.9259 | 0.9490 | 0.9942 | 0.9971 | 0.9875 | 1014 |
| other | 0.7838 | 0.9542 | 0.7571 | 0.7703 | 0.9604 | 0.9561 | 0.8607 | 879 |
| stroma | 0.7484 | 0.9606 | 0.7444 | 0.7464 | 0.9613 | 0.9628 | 0.7880 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=79 FP=34 TN=5568 FN=21
- **cancer**: TP=1121 FP=112 TN=4261 FN=208
- **empty**: TP=1609 FP=8 TN=4073 FN=12
- **normal**: TP=987 FP=79 TN=4609 FN=27
- **other**: TP=689 FP=221 TN=4602 FN=190
- **stroma**: TP=568 FP=195 TN=4748 FN=191

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 79 | 2 | 2 | 7 | 6 | 4 |
| cancer | 29 | 1121 | 0 | 40 | 72 | 67 |
| empty | 0 | 0 | 1609 | 0 | 10 | 2 |
| normal | 2 | 18 | 0 | 987 | 1 | 6 |
| other | 2 | 57 | 6 | 9 | 689 | 116 |
| stroma | 1 | 35 | 0 | 23 | 132 | 568 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6991    0.7900    0.7418       100
      cancer     0.9092    0.8435    0.8751      1329
       empty     0.9951    0.9926    0.9938      1621
      normal     0.9259    0.9734    0.9490      1014
       other     0.7571    0.7838    0.7703       879
      stroma     0.7444    0.7484    0.7464       759

    accuracy                         0.8862      5702
   macro avg     0.8385    0.8553    0.8461      5702
weighted avg     0.8875    0.8862    0.8864      5702

```