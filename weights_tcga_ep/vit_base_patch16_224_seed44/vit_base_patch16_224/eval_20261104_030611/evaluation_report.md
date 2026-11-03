# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/vit_base_patch16_224_seed44/vit_base_patch16_224/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8315 |
| Accuracy | 0.8702 |
| Macro-AUROC (OvR) | 0.9751 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8720 |
| Balanced Acc | 0.8542 |
| Macro-Sensitivity | 0.8542 |
| Macro-Specificity | 0.9745 |
| Macro-Precision | 0.8206 |
| AUC-PR (Macro) | 0.8985 |
| Cohen's Kappa | 0.8369 |
| MCC | 0.8388 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 358.2 (2.79 ms/张) |
| End-to-End FPS | 230.3 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8400 | 0.9916 | 0.6412 | 0.7273 | 0.9971 | 0.9937 | 0.8623 | 100 |
| cancer | 0.7713 | 0.9865 | 0.9456 | 0.8496 | 0.9342 | 0.9610 | 0.9331 | 1329 |
| empty | 0.9963 | 0.9924 | 0.9812 | 0.9887 | 0.9985 | 0.9998 | 0.9996 | 1621 |
| normal | 0.9359 | 0.9834 | 0.9241 | 0.9299 | 0.9861 | 0.9953 | 0.9805 | 1014 |
| other | 0.7372 | 0.9621 | 0.7798 | 0.7579 | 0.9526 | 0.9409 | 0.8078 | 879 |
| stroma | 0.8445 | 0.9308 | 0.6521 | 0.7359 | 0.9750 | 0.9599 | 0.8079 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=84 FP=47 TN=5555 FN=16
- **cancer**: TP=1025 FP=59 TN=4314 FN=304
- **empty**: TP=1615 FP=31 TN=4050 FN=6
- **normal**: TP=949 FP=78 TN=4610 FN=65
- **other**: TP=648 FP=183 TN=4640 FN=231
- **stroma**: TP=641 FP=342 TN=4601 FN=118

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 84 | 1 | 3 | 4 | 6 | 2 |
| cancer | 35 | 1025 | 2 | 41 | 73 | 153 |
| empty | 1 | 0 | 1615 | 0 | 3 | 2 |
| normal | 2 | 15 | 0 | 949 | 27 | 21 |
| other | 2 | 28 | 26 | 11 | 648 | 164 |
| stroma | 7 | 15 | 0 | 22 | 74 | 641 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6412    0.8400    0.7273       100
      cancer     0.9456    0.7713    0.8496      1329
       empty     0.9812    0.9963    0.9887      1621
      normal     0.9241    0.9359    0.9299      1014
       other     0.7798    0.7372    0.7579       879
      stroma     0.6521    0.8445    0.7359       759

    accuracy                         0.8702      5702
   macro avg     0.8206    0.8542    0.8315      5702
weighted avg     0.8819    0.8702    0.8720      5702

```