# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_egb_scg_seed43/convnext_egb_scg/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8569 |
| Accuracy | 0.8904 |
| Macro-AUROC (OvR) | 0.9806 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8906 |
| Balanced Acc | 0.8644 |
| Macro-Sensitivity | 0.8644 |
| Macro-Specificity | 0.9782 |
| Macro-Precision | 0.8601 |
| AUC-PR (Macro) | 0.9037 |
| Cohen's Kappa | 0.8618 |
| MCC | 0.8636 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.29M |
| Pure-Inference FPS | 718.2 (1.39 ms/张) |
| End-to-End FPS | 346.9 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8100 | 0.9954 | 0.7570 | 0.7826 | 0.9966 | 0.9959 | 0.8577 | 100 |
| cancer | 0.8608 | 0.9684 | 0.8924 | 0.8763 | 0.9581 | 0.9726 | 0.9328 | 1329 |
| empty | 0.9951 | 0.9975 | 0.9938 | 0.9945 | 0.9980 | 0.9999 | 0.9996 | 1621 |
| normal | 0.9596 | 0.9863 | 0.9383 | 0.9488 | 0.9912 | 0.9969 | 0.9866 | 1014 |
| other | 0.6769 | 0.9871 | 0.9056 | 0.7747 | 0.9437 | 0.9515 | 0.8688 | 879 |
| stroma | 0.8841 | 0.9343 | 0.6737 | 0.7647 | 0.9813 | 0.9668 | 0.7764 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=81 FP=26 TN=5576 FN=19
- **cancer**: TP=1144 FP=138 TN=4235 FN=185
- **empty**: TP=1613 FP=10 TN=4071 FN=8
- **normal**: TP=973 FP=64 TN=4624 FN=41
- **other**: TP=595 FP=62 TN=4761 FN=284
- **stroma**: TP=671 FP=325 TN=4618 FN=88

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 81 | 2 | 0 | 8 | 5 | 4 |
| cancer | 20 | 1144 | 1 | 31 | 24 | 109 |
| empty | 0 | 0 | 1613 | 0 | 6 | 2 |
| normal | 3 | 15 | 0 | 973 | 1 | 22 |
| other | 3 | 73 | 9 | 11 | 595 | 188 |
| stroma | 0 | 48 | 0 | 14 | 26 | 671 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7570    0.8100    0.7826       100
      cancer     0.8924    0.8608    0.8763      1329
       empty     0.9938    0.9951    0.9945      1621
      normal     0.9383    0.9596    0.9488      1014
       other     0.9056    0.6769    0.7747       879
      stroma     0.6737    0.8841    0.7647       759

    accuracy                         0.8904      5702
   macro avg     0.8601    0.8644    0.8569      5702
weighted avg     0.8999    0.8904    0.8906      5702

```