# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/vit_base_patch16_224_seed43/vit_base_patch16_224/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7882 |
| Accuracy | 0.8340 |
| Macro-AUROC (OvR) | 0.9540 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8291 |
| Balanced Acc | 0.8098 |
| Macro-Sensitivity | 0.8098 |
| Macro-Specificity | 0.9656 |
| Macro-Precision | 0.7804 |
| AUC-PR (Macro) | 0.8432 |
| Cohen's Kappa | 0.7862 |
| MCC | 0.7873 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 315.0 (3.17 ms/张) |
| End-to-End FPS | 213.2 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.9598 | 0.9901 | 0.6497 | 0.7748 | 0.9992 | 0.9974 | 0.8638 | 199 |
| cancer | 0.9191 | 0.9528 | 0.8657 | 0.8916 | 0.9727 | 0.9862 | 0.9682 | 2645 |
| empty | 0.9966 | 0.9597 | 0.9161 | 0.9547 | 0.9984 | 0.9849 | 0.9314 | 3256 |
| normal | 0.7331 | 0.9756 | 0.8543 | 0.7891 | 0.9494 | 0.9548 | 0.8707 | 1735 |
| other | 0.5946 | 0.9433 | 0.6686 | 0.6295 | 0.9236 | 0.8554 | 0.6600 | 1717 |
| stroma | 0.6553 | 0.9723 | 0.7279 | 0.6897 | 0.9614 | 0.9453 | 0.7652 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=191 FP=103 TN=10332 FN=8
- **cancer**: TP=2431 FP=377 TN=7612 FN=214
- **empty**: TP=3245 FP=297 TN=7081 FN=11
- **normal**: TP=1272 FP=217 TN=8682 FN=463
- **other**: TP=1021 FP=506 TN=8411 FN=696
- **stroma**: TP=709 FP=265 TN=9287 FN=373

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 191 | 2 | 2 | 3 | 0 | 1 |
| cancer | 18 | 2431 | 1 | 110 | 72 | 13 |
| empty | 0 | 0 | 3245 | 0 | 11 | 0 |
| normal | 9 | 124 | 1 | 1272 | 205 | 124 |
| other | 69 | 181 | 293 | 26 | 1021 | 127 |
| stroma | 7 | 70 | 0 | 78 | 218 | 709 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6497    0.9598    0.7748       199
      cancer     0.8657    0.9191    0.8916      2645
       empty     0.9161    0.9966    0.9547      3256
      normal     0.8543    0.7331    0.7891      1735
       other     0.6686    0.5946    0.6295      1717
      stroma     0.7279    0.6553    0.6897      1082

    accuracy                         0.8340     10634
   macro avg     0.7804    0.8098    0.7882     10634
weighted avg     0.8294    0.8340    0.8291     10634

```