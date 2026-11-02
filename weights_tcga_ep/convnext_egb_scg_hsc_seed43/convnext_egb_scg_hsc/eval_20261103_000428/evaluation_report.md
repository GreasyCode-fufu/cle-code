# convnext_egb_scg_hsc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_egb_scg_hsc_seed43/convnext_egb_scg_hsc/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8103 |
| Accuracy | 0.8478 |
| Macro-AUROC (OvR) | 0.9641 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8433 |
| Balanced Acc | 0.8109 |
| Macro-Sensitivity | 0.8109 |
| Macro-Specificity | 0.9684 |
| Macro-Precision | 0.8134 |
| AUC-PR (Macro) | 0.8795 |
| Cohen's Kappa | 0.8038 |
| MCC | 0.8047 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.58M |
| Pure-Inference FPS | 515.4 (1.94 ms/张) |
| End-to-End FPS | 289.8 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8894 | 0.9959 | 0.8045 | 0.8449 | 0.9979 | 0.9958 | 0.9331 | 199 |
| cancer | 0.9380 | 0.9573 | 0.8792 | 0.9076 | 0.9790 | 0.9854 | 0.9686 | 2645 |
| empty | 0.9969 | 0.9615 | 0.9195 | 0.9567 | 0.9986 | 0.9842 | 0.9614 | 3256 |
| normal | 0.7775 | 0.9788 | 0.8771 | 0.8243 | 0.9576 | 0.9772 | 0.9137 | 1735 |
| other | 0.6238 | 0.9468 | 0.6932 | 0.6567 | 0.9289 | 0.8866 | 0.7288 | 1717 |
| stroma | 0.6396 | 0.9700 | 0.7068 | 0.6715 | 0.9596 | 0.9553 | 0.7713 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=177 FP=43 TN=10392 FN=22
- **cancer**: TP=2481 FP=341 TN=7648 FN=164
- **empty**: TP=3246 FP=284 TN=7094 FN=10
- **normal**: TP=1349 FP=189 TN=8710 FN=386
- **other**: TP=1071 FP=474 TN=8443 FN=646
- **stroma**: TP=692 FP=287 TN=9265 FN=390

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 177 | 5 | 2 | 4 | 9 | 2 |
| cancer | 2 | 2481 | 2 | 68 | 34 | 58 |
| empty | 0 | 2 | 3246 | 0 | 8 | 0 |
| normal | 2 | 87 | 0 | 1349 | 183 | 114 |
| other | 35 | 183 | 280 | 35 | 1071 | 113 |
| stroma | 4 | 64 | 0 | 82 | 240 | 692 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8045    0.8894    0.8449       199
      cancer     0.8792    0.9380    0.9076      2645
       empty     0.9195    0.9969    0.9567      3256
      normal     0.8771    0.7775    0.8243      1735
       other     0.6932    0.6238    0.6567      1717
      stroma     0.7068    0.6396    0.6715      1082

    accuracy                         0.8478     10634
   macro avg     0.8134    0.8109    0.8103     10634
weighted avg     0.8422    0.8478    0.8433     10634

```