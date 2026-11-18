# cle_convnext_sn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/cle_convnext_sn_seed42/cle_convnext_sn/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8642 |
| Accuracy | 0.8944 |
| Macro-AUROC (OvR) | 0.9753 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8955 |
| Balanced Acc | 0.8633 |
| Macro-Sensitivity | 0.8633 |
| Macro-Specificity | 0.9790 |
| Macro-Precision | 0.8757 |
| AUC-PR (Macro) | 0.9098 |
| Cohen's Kappa | 0.8668 |
| MCC | 0.8686 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 628.4 (1.59 ms/张) |
| End-to-End FPS | 321.7 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7800 | 0.9971 | 0.8298 | 0.8041 | 0.9961 | 0.9972 | 0.8895 | 100 |
| cancer | 0.8736 | 0.9714 | 0.9028 | 0.8880 | 0.9620 | 0.9793 | 0.9525 | 1329 |
| empty | 0.9951 | 0.9975 | 0.9938 | 0.9945 | 0.9980 | 0.9999 | 0.9997 | 1621 |
| normal | 0.9517 | 0.9870 | 0.9405 | 0.9461 | 0.9895 | 0.9945 | 0.9827 | 1014 |
| other | 0.7019 | 0.9896 | 0.9250 | 0.7982 | 0.9480 | 0.9146 | 0.8570 | 879 |
| stroma | 0.8775 | 0.9312 | 0.6620 | 0.7547 | 0.9802 | 0.9661 | 0.7776 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=78 FP=16 TN=5586 FN=22
- **cancer**: TP=1161 FP=125 TN=4248 FN=168
- **empty**: TP=1613 FP=10 TN=4071 FN=8
- **normal**: TP=965 FP=61 TN=4627 FN=49
- **other**: TP=617 FP=50 TN=4773 FN=262
- **stroma**: TP=666 FP=340 TN=4603 FN=93

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 78 | 7 | 1 | 6 | 5 | 3 |
| cancer | 8 | 1161 | 2 | 11 | 15 | 132 |
| empty | 1 | 0 | 1613 | 0 | 6 | 1 |
| normal | 2 | 13 | 0 | 965 | 1 | 33 |
| other | 3 | 64 | 7 | 17 | 617 | 171 |
| stroma | 2 | 41 | 0 | 27 | 23 | 666 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8298    0.7800    0.8041       100
      cancer     0.9028    0.8736    0.8880      1329
       empty     0.9938    0.9951    0.9945      1621
      normal     0.9405    0.9517    0.9461      1014
       other     0.9250    0.7019    0.7982       879
      stroma     0.6620    0.8775    0.7547       759

    accuracy                         0.8944      5702
   macro avg     0.8757    0.8633    0.8642      5702
weighted avg     0.9055    0.8944    0.8955      5702

```