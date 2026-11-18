# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce4s_seed44/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8416 |
| Accuracy | 0.8736 |
| Macro-AUROC (OvR) | 0.9757 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8760 |
| Balanced Acc | 0.8388 |
| Macro-Sensitivity | 0.8388 |
| Macro-Specificity | 0.9748 |
| Macro-Precision | 0.8484 |
| AUC-PR (Macro) | 0.8840 |
| Cohen's Kappa | 0.8405 |
| MCC | 0.8412 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9127.7 (0.11 ms/张) |
| End-to-End FPS | 548.0 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7600 | 0.9970 | 0.8172 | 0.7876 | 0.9957 | 0.9932 | 0.8382 | 100 |
| cancer | 0.8661 | 0.9675 | 0.8902 | 0.8780 | 0.9596 | 0.9768 | 0.9472 | 1329 |
| empty | 0.9951 | 0.9971 | 0.9926 | 0.9938 | 0.9980 | 0.9999 | 0.9999 | 1621 |
| normal | 0.8629 | 0.9921 | 0.9594 | 0.9086 | 0.9710 | 0.9890 | 0.9697 | 1014 |
| other | 0.7543 | 0.9629 | 0.7874 | 0.7705 | 0.9556 | 0.9459 | 0.8597 | 879 |
| stroma | 0.7945 | 0.9324 | 0.6435 | 0.7111 | 0.9673 | 0.9496 | 0.6893 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=76 FP=17 TN=5585 FN=24
- **cancer**: TP=1151 FP=142 TN=4231 FN=178
- **empty**: TP=1613 FP=12 TN=4069 FN=8
- **normal**: TP=875 FP=37 TN=4651 FN=139
- **other**: TP=663 FP=179 TN=4644 FN=216
- **stroma**: TP=603 FP=334 TN=4609 FN=156

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 76 | 4 | 1 | 3 | 9 | 7 |
| cancer | 9 | 1151 | 1 | 13 | 50 | 105 |
| empty | 0 | 0 | 1613 | 0 | 6 | 2 |
| normal | 4 | 37 | 0 | 875 | 36 | 62 |
| other | 2 | 40 | 9 | 7 | 663 | 158 |
| stroma | 2 | 61 | 1 | 14 | 78 | 603 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8172    0.7600    0.7876       100
      cancer     0.8902    0.8661    0.8780      1329
       empty     0.9926    0.9951    0.9938      1621
      normal     0.9594    0.8629    0.9086      1014
       other     0.7874    0.7543    0.7705       879
      stroma     0.6435    0.7945    0.7111       759

    accuracy                         0.8736      5702
   macro avg     0.8484    0.8388    0.8416      5702
weighted avg     0.8817    0.8736    0.8760      5702

```