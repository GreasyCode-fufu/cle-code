# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_r6_fz16_seed42/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8182 |
| Accuracy | 0.8486 |
| Macro-AUROC (OvR) | 0.9721 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8500 |
| Balanced Acc | 0.8242 |
| Macro-Sensitivity | 0.8242 |
| Macro-Specificity | 0.9695 |
| Macro-Precision | 0.8175 |
| AUC-PR (Macro) | 0.8746 |
| Cohen's Kappa | 0.8087 |
| MCC | 0.8097 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.92M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 7062.1 (0.14 ms/张) |
| End-to-End FPS | 190.7 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8600 | 0.9950 | 0.7544 | 0.8037 | 0.9975 | 0.9924 | 0.8501 | 100 |
| cancer | 0.8969 | 0.9337 | 0.8043 | 0.8481 | 0.9675 | 0.9768 | 0.9392 | 1329 |
| empty | 0.9889 | 0.9998 | 0.9994 | 0.9941 | 0.9956 | 0.9995 | 0.9992 | 1621 |
| normal | 0.7880 | 0.9900 | 0.9444 | 0.8591 | 0.9557 | 0.9878 | 0.9550 | 1014 |
| other | 0.7292 | 0.9637 | 0.7855 | 0.7563 | 0.9513 | 0.9412 | 0.8480 | 879 |
| stroma | 0.6825 | 0.9349 | 0.6167 | 0.6479 | 0.9504 | 0.9351 | 0.6561 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=86 FP=28 TN=5574 FN=14
- **cancer**: TP=1192 FP=290 TN=4083 FN=137
- **empty**: TP=1603 FP=1 TN=4080 FN=18
- **normal**: TP=799 FP=47 TN=4641 FN=215
- **other**: TP=641 FP=175 TN=4648 FN=238
- **stroma**: TP=518 FP=322 TN=4621 FN=241

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 86 | 3 | 0 | 2 | 4 | 5 |
| cancer | 17 | 1192 | 0 | 15 | 33 | 72 |
| empty | 0 | 0 | 1603 | 1 | 15 | 2 |
| normal | 2 | 76 | 0 | 799 | 36 | 101 |
| other | 3 | 78 | 1 | 14 | 641 | 142 |
| stroma | 6 | 133 | 0 | 15 | 87 | 518 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7544    0.8600    0.8037       100
      cancer     0.8043    0.8969    0.8481      1329
       empty     0.9994    0.9889    0.9941      1621
      normal     0.9444    0.7880    0.8591      1014
       other     0.7855    0.7292    0.7563       879
      stroma     0.6167    0.6825    0.6479       759

    accuracy                         0.8486      5702
   macro avg     0.8175    0.8242    0.8182      5702
weighted avg     0.8559    0.8486    0.8500      5702

```