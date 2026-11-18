# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_dce4m_seed42/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7945 |
| Accuracy | 0.8394 |
| Macro-AUROC (OvR) | 0.9580 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8350 |
| Balanced Acc | 0.7951 |
| Macro-Sensitivity | 0.7951 |
| Macro-Specificity | 0.9665 |
| Macro-Precision | 0.7998 |
| AUC-PR (Macro) | 0.8686 |
| Cohen's Kappa | 0.7928 |
| MCC | 0.7942 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9211.0 (0.11 ms/张) |
| End-to-End FPS | 558.4 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8191 | 0.9943 | 0.7342 | 0.7743 | 0.9965 | 0.9945 | 0.8993 | 199 |
| cancer | 0.9270 | 0.9437 | 0.8449 | 0.8841 | 0.9750 | 0.9838 | 0.9517 | 2645 |
| empty | 0.9963 | 0.9606 | 0.9177 | 0.9554 | 0.9983 | 0.9945 | 0.9867 | 3256 |
| normal | 0.7239 | 0.9836 | 0.8959 | 0.8008 | 0.9481 | 0.9327 | 0.8684 | 1735 |
| other | 0.6296 | 0.9500 | 0.7079 | 0.6665 | 0.9302 | 0.9076 | 0.7507 | 1717 |
| stroma | 0.6747 | 0.9669 | 0.6979 | 0.6861 | 0.9633 | 0.9347 | 0.7549 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=163 FP=59 TN=10376 FN=36
- **cancer**: TP=2452 FP=450 TN=7539 FN=193
- **empty**: TP=3244 FP=291 TN=7087 FN=12
- **normal**: TP=1256 FP=146 TN=8753 FN=479
- **other**: TP=1081 FP=446 TN=8471 FN=636
- **stroma**: TP=730 FP=316 TN=9236 FN=352

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 163 | 5 | 2 | 2 | 26 | 1 |
| cancer | 8 | 2452 | 1 | 57 | 87 | 40 |
| empty | 0 | 0 | 3244 | 0 | 12 | 0 |
| normal | 4 | 161 | 3 | 1256 | 144 | 167 |
| other | 45 | 181 | 285 | 17 | 1081 | 108 |
| stroma | 2 | 103 | 0 | 70 | 177 | 730 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7342    0.8191    0.7743       199
      cancer     0.8449    0.9270    0.8841      2645
       empty     0.9177    0.9963    0.9554      3256
      normal     0.8959    0.7239    0.8008      1735
       other     0.7079    0.6296    0.6665      1717
      stroma     0.6979    0.6747    0.6861      1082

    accuracy                         0.8394     10634
   macro avg     0.7998    0.7951    0.7945     10634
weighted avg     0.8364    0.8394    0.8350     10634

```