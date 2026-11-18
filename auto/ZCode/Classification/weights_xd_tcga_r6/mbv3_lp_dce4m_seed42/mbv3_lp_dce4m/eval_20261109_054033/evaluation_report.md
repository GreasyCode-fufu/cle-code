# mbv3_lp_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_lp_dce4m_seed42/mbv3_lp_dce4m/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7804 |
| Accuracy | 0.8265 |
| Macro-AUROC (OvR) | 0.9630 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8211 |
| Balanced Acc | 0.7730 |
| Macro-Sensitivity | 0.7730 |
| Macro-Specificity | 0.9638 |
| Macro-Precision | 0.7926 |
| AUC-PR (Macro) | 0.8570 |
| Cohen's Kappa | 0.7759 |
| MCC | 0.7771 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 7099.3 (0.14 ms/张) |
| End-to-End FPS | 22.2 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7940 | 0.9961 | 0.7940 | 0.7940 | 0.9961 | 0.9947 | 0.8925 | 199 |
| cancer | 0.9195 | 0.9358 | 0.8258 | 0.8701 | 0.9723 | 0.9837 | 0.9563 | 2645 |
| empty | 0.9951 | 0.9630 | 0.9223 | 0.9573 | 0.9978 | 0.9860 | 0.9636 | 3256 |
| normal | 0.7412 | 0.9769 | 0.8619 | 0.7970 | 0.9509 | 0.9602 | 0.8751 | 1735 |
| other | 0.6098 | 0.9412 | 0.6665 | 0.6369 | 0.9261 | 0.9011 | 0.7238 | 1717 |
| stroma | 0.5786 | 0.9698 | 0.6849 | 0.6273 | 0.9531 | 0.9526 | 0.7309 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=158 FP=41 TN=10394 FN=41
- **cancer**: TP=2432 FP=513 TN=7476 FN=213
- **empty**: TP=3240 FP=273 TN=7105 FN=16
- **normal**: TP=1286 FP=206 TN=8693 FN=449
- **other**: TP=1047 FP=524 TN=8393 FN=670
- **stroma**: TP=626 FP=288 TN=9264 FN=456

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 158 | 5 | 1 | 6 | 25 | 4 |
| cancer | 7 | 2432 | 0 | 70 | 107 | 29 |
| empty | 0 | 0 | 3240 | 0 | 16 | 0 |
| normal | 4 | 175 | 0 | 1286 | 140 | 130 |
| other | 29 | 218 | 272 | 26 | 1047 | 125 |
| stroma | 1 | 115 | 0 | 104 | 236 | 626 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7940    0.7940    0.7940       199
      cancer     0.8258    0.9195    0.8701      2645
       empty     0.9223    0.9951    0.9573      3256
      normal     0.8619    0.7412    0.7970      1735
       other     0.6665    0.6098    0.6369      1717
      stroma     0.6849    0.5786    0.6273      1082

    accuracy                         0.8265     10634
   macro avg     0.7926    0.7730    0.7804     10634
weighted avg     0.8206    0.8265    0.8211     10634

```