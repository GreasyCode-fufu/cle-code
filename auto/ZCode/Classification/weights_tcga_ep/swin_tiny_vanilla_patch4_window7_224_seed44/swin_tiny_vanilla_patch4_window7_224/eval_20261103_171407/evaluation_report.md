# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `weights_tcga_ep/swin_tiny_vanilla_patch4_window7_224_seed44/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8342 |
| Accuracy | 0.8630 |
| Macro-AUROC (OvR) | 0.9719 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8664 |
| Balanced Acc | 0.8425 |
| Macro-Sensitivity | 0.8425 |
| Macro-Specificity | 0.9732 |
| Macro-Precision | 0.8364 |
| AUC-PR (Macro) | 0.8906 |
| Cohen's Kappa | 0.8282 |
| MCC | 0.8309 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 1620.0 (0.62 ms/张) |
| End-to-End FPS | 465.1 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7800 | 0.9961 | 0.7800 | 0.7800 | 0.9961 | 0.9852 | 0.8545 | 100 |
| cancer | 0.7254 | 0.9872 | 0.9451 | 0.8208 | 0.9220 | 0.9578 | 0.9037 | 1329 |
| empty | 0.9870 | 0.9998 | 0.9994 | 0.9932 | 0.9949 | 0.9998 | 0.9996 | 1621 |
| normal | 0.9428 | 0.9904 | 0.9550 | 0.9489 | 0.9877 | 0.9949 | 0.9854 | 1014 |
| other | 0.7793 | 0.9430 | 0.7135 | 0.7450 | 0.9591 | 0.9388 | 0.8512 | 879 |
| stroma | 0.8406 | 0.9227 | 0.6255 | 0.7173 | 0.9742 | 0.9547 | 0.7492 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=78 FP=22 TN=5580 FN=22
- **cancer**: TP=964 FP=56 TN=4317 FN=365
- **empty**: TP=1600 FP=1 TN=4080 FN=21
- **normal**: TP=956 FP=45 TN=4643 FN=58
- **other**: TP=685 FP=275 TN=4548 FN=194
- **stroma**: TP=638 FP=382 TN=4561 FN=121

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 78 | 0 | 0 | 4 | 12 | 6 |
| cancer | 15 | 964 | 0 | 16 | 136 | 198 |
| empty | 0 | 0 | 1600 | 0 | 19 | 2 |
| normal | 1 | 9 | 0 | 956 | 24 | 24 |
| other | 0 | 33 | 1 | 8 | 685 | 152 |
| stroma | 6 | 14 | 0 | 17 | 84 | 638 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7800    0.7800    0.7800       100
      cancer     0.9451    0.7254    0.8208      1329
       empty     0.9994    0.9870    0.9932      1621
      normal     0.9550    0.9428    0.9489      1014
       other     0.7135    0.7793    0.7450       879
      stroma     0.6255    0.8406    0.7173       759

    accuracy                         0.8630      5702
   macro avg     0.8364    0.8425    0.8342      5702
weighted avg     0.8812    0.8630    0.8664      5702

```