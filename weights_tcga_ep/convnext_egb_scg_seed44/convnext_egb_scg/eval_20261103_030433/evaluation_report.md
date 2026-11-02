# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_egb_scg_seed44/convnext_egb_scg/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8173 |
| Accuracy | 0.8602 |
| Macro-AUROC (OvR) | 0.9788 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8596 |
| Balanced Acc | 0.8480 |
| Macro-Sensitivity | 0.8480 |
| Macro-Specificity | 0.9725 |
| Macro-Precision | 0.8065 |
| AUC-PR (Macro) | 0.8946 |
| Cohen's Kappa | 0.8246 |
| MCC | 0.8279 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.29M |
| Pure-Inference FPS | 714.2 (1.40 ms/张) |
| End-to-End FPS | 346.1 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8300 | 0.9895 | 0.5845 | 0.6860 | 0.9969 | 0.9905 | 0.8160 | 100 |
| cancer | 0.6945 | 0.9835 | 0.9276 | 0.7943 | 0.9137 | 0.9669 | 0.9187 | 1329 |
| empty | 0.9975 | 0.9951 | 0.9878 | 0.9926 | 0.9990 | 1.0000 | 0.9999 | 1621 |
| normal | 0.9684 | 0.9610 | 0.8429 | 0.9013 | 0.9929 | 0.9948 | 0.9833 | 1014 |
| other | 0.7281 | 0.9733 | 0.8322 | 0.7767 | 0.9516 | 0.9576 | 0.8663 | 879 |
| stroma | 0.8696 | 0.9324 | 0.6640 | 0.7530 | 0.9790 | 0.9629 | 0.7833 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=83 FP=59 TN=5543 FN=17
- **cancer**: TP=923 FP=72 TN=4301 FN=406
- **empty**: TP=1617 FP=20 TN=4061 FN=4
- **normal**: TP=982 FP=183 TN=4505 FN=32
- **other**: TP=640 FP=129 TN=4694 FN=239
- **stroma**: TP=660 FP=334 TN=4609 FN=99

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 83 | 0 | 7 | 5 | 4 | 1 |
| cancer | 42 | 923 | 0 | 143 | 72 | 149 |
| empty | 0 | 0 | 1617 | 0 | 4 | 0 |
| normal | 2 | 4 | 0 | 982 | 1 | 25 |
| other | 6 | 49 | 13 | 12 | 640 | 159 |
| stroma | 9 | 19 | 0 | 23 | 48 | 660 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.5845    0.8300    0.6860       100
      cancer     0.9276    0.6945    0.7943      1329
       empty     0.9878    0.9975    0.9926      1621
      normal     0.8429    0.9684    0.9013      1014
       other     0.8322    0.7281    0.7767       879
      stroma     0.6640    0.8696    0.7530       759

    accuracy                         0.8602      5702
   macro avg     0.8065    0.8480    0.8173      5702
weighted avg     0.8739    0.8602    0.8596      5702

```