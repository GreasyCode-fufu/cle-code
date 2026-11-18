# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_abl/convnext_v2m_seed44/convnext_v2m/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8569 |
| Accuracy | 0.8841 |
| Macro-AUROC (OvR) | 0.9801 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8844 |
| Balanced Acc | 0.8617 |
| Macro-Sensitivity | 0.8617 |
| Macro-Specificity | 0.9767 |
| Macro-Precision | 0.8538 |
| AUC-PR (Macro) | 0.9068 |
| Cohen's Kappa | 0.8536 |
| MCC | 0.8538 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 635.4 (1.57 ms/张) |
| End-to-End FPS | 324.6 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8700 | 0.9963 | 0.8056 | 0.8365 | 0.9977 | 0.9982 | 0.9149 | 100 |
| cancer | 0.8849 | 0.9598 | 0.8698 | 0.8773 | 0.9648 | 0.9760 | 0.9393 | 1329 |
| empty | 0.9963 | 0.9975 | 0.9938 | 0.9951 | 0.9985 | 1.0000 | 0.9999 | 1621 |
| normal | 0.9379 | 0.9908 | 0.9567 | 0.9472 | 0.9866 | 0.9959 | 0.9856 | 1014 |
| other | 0.7327 | 0.9695 | 0.8142 | 0.7713 | 0.9521 | 0.9536 | 0.8599 | 879 |
| stroma | 0.7484 | 0.9466 | 0.6827 | 0.7140 | 0.9608 | 0.9572 | 0.7411 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=87 FP=21 TN=5581 FN=13
- **cancer**: TP=1176 FP=176 TN=4197 FN=153
- **empty**: TP=1615 FP=10 TN=4071 FN=6
- **normal**: TP=951 FP=43 TN=4645 FN=63
- **other**: TP=644 FP=147 TN=4676 FN=235
- **stroma**: TP=568 FP=264 TN=4679 FN=191

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 87 | 3 | 0 | 1 | 4 | 5 |
| cancer | 11 | 1176 | 1 | 13 | 38 | 90 |
| empty | 1 | 0 | 1615 | 0 | 4 | 1 |
| normal | 3 | 30 | 0 | 951 | 6 | 24 |
| other | 5 | 65 | 8 | 13 | 644 | 144 |
| stroma | 1 | 78 | 1 | 16 | 95 | 568 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8056    0.8700    0.8365       100
      cancer     0.8698    0.8849    0.8773      1329
       empty     0.9938    0.9963    0.9951      1621
      normal     0.9567    0.9379    0.9472      1014
       other     0.8142    0.7327    0.7713       879
      stroma     0.6827    0.7484    0.7140       759

    accuracy                         0.8841      5702
   macro avg     0.8538    0.8617    0.8569      5702
weighted avg     0.8859    0.8841    0.8844      5702

```