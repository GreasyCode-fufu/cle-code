# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_v2bc_seed44/convnext_v2bc/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8573 |
| Accuracy | 0.8951 |
| Macro-AUROC (OvR) | 0.9806 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8962 |
| Balanced Acc | 0.8698 |
| Macro-Sensitivity | 0.8698 |
| Macro-Specificity | 0.9793 |
| Macro-Precision | 0.8476 |
| AUC-PR (Macro) | 0.9111 |
| Cohen's Kappa | 0.8677 |
| MCC | 0.8679 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 532.7 (1.88 ms/张) |
| End-to-End FPS | 295.6 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8400 | 0.9936 | 0.7000 | 0.7636 | 0.9971 | 0.9937 | 0.8650 | 100 |
| cancer | 0.8894 | 0.9762 | 0.9191 | 0.9040 | 0.9667 | 0.9805 | 0.9584 | 1329 |
| empty | 0.9938 | 0.9985 | 0.9963 | 0.9951 | 0.9976 | 0.9998 | 0.9997 | 1621 |
| normal | 0.9359 | 0.9898 | 0.9519 | 0.9438 | 0.9862 | 0.9964 | 0.9858 | 1014 |
| other | 0.7861 | 0.9668 | 0.8120 | 0.7988 | 0.9612 | 0.9514 | 0.8699 | 879 |
| stroma | 0.7734 | 0.9506 | 0.7064 | 0.7384 | 0.9647 | 0.9615 | 0.7876 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=84 FP=36 TN=5566 FN=16
- **cancer**: TP=1182 FP=104 TN=4269 FN=147
- **empty**: TP=1611 FP=6 TN=4075 FN=10
- **normal**: TP=949 FP=48 TN=4640 FN=65
- **other**: TP=691 FP=160 TN=4663 FN=188
- **stroma**: TP=587 FP=244 TN=4699 FN=172

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 84 | 2 | 1 | 5 | 7 | 1 |
| cancer | 8 | 1182 | 1 | 17 | 25 | 96 |
| empty | 0 | 0 | 1611 | 0 | 8 | 2 |
| normal | 5 | 10 | 0 | 949 | 20 | 30 |
| other | 7 | 52 | 4 | 10 | 691 | 115 |
| stroma | 16 | 40 | 0 | 16 | 100 | 587 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7000    0.8400    0.7636       100
      cancer     0.9191    0.8894    0.9040      1329
       empty     0.9963    0.9938    0.9951      1621
      normal     0.9519    0.9359    0.9438      1014
       other     0.8120    0.7861    0.7988       879
      stroma     0.7064    0.7734    0.7384       759

    accuracy                         0.8951      5702
   macro avg     0.8476    0.8698    0.8573      5702
weighted avg     0.8982    0.8951    0.8962      5702

```