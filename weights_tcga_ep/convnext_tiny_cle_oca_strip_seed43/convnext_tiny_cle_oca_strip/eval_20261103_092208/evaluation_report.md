# convnext_tiny_cle_oca_strip 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_tiny_cle_oca_strip_seed43/convnext_tiny_cle_oca_strip/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8529 |
| Accuracy | 0.8864 |
| Macro-AUROC (OvR) | 0.9758 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8890 |
| Balanced Acc | 0.8683 |
| Macro-Sensitivity | 0.8683 |
| Macro-Specificity | 0.9778 |
| Macro-Precision | 0.8507 |
| AUC-PR (Macro) | 0.8933 |
| Cohen's Kappa | 0.8572 |
| MCC | 0.8599 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.47G |
| FLOPs | 8.94G |
| Params | 27.84M |
| Pure-Inference FPS | 807.8 (1.24 ms/张) |
| End-to-End FPS | 365.2 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8300 | 0.9941 | 0.7155 | 0.7685 | 0.9970 | 0.9957 | 0.8737 | 100 |
| cancer | 0.8059 | 0.9899 | 0.9605 | 0.8764 | 0.9438 | 0.9688 | 0.9395 | 1329 |
| empty | 0.9957 | 0.9985 | 0.9963 | 0.9960 | 0.9983 | 1.0000 | 0.9999 | 1621 |
| normal | 0.9448 | 0.9817 | 0.9176 | 0.9310 | 0.9880 | 0.9923 | 0.9772 | 1014 |
| other | 0.7338 | 0.9811 | 0.8764 | 0.7988 | 0.9529 | 0.9380 | 0.8633 | 879 |
| stroma | 0.8999 | 0.9215 | 0.6377 | 0.7464 | 0.9836 | 0.9601 | 0.7065 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=83 FP=33 TN=5569 FN=17
- **cancer**: TP=1071 FP=44 TN=4329 FN=258
- **empty**: TP=1614 FP=6 TN=4075 FN=7
- **normal**: TP=958 FP=86 TN=4602 FN=56
- **other**: TP=645 FP=91 TN=4732 FN=234
- **stroma**: TP=683 FP=388 TN=4555 FN=76

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 83 | 1 | 0 | 2 | 7 | 7 |
| cancer | 26 | 1071 | 1 | 38 | 41 | 152 |
| empty | 0 | 0 | 1614 | 0 | 4 | 3 |
| normal | 2 | 8 | 0 | 958 | 3 | 43 |
| other | 3 | 26 | 5 | 17 | 645 | 183 |
| stroma | 2 | 9 | 0 | 29 | 36 | 683 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7155    0.8300    0.7685       100
      cancer     0.9605    0.8059    0.8764      1329
       empty     0.9963    0.9957    0.9960      1621
      normal     0.9176    0.9448    0.9310      1014
       other     0.8764    0.7338    0.7988       879
      stroma     0.6377    0.8999    0.7464       759

    accuracy                         0.8864      5702
   macro avg     0.8507    0.8683    0.8529      5702
weighted avg     0.9028    0.8864    0.8890      5702

```