# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/fast_t0_seed42/fast_t0/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7842 |
| Accuracy | 0.8278 |
| Macro-AUROC (OvR) | 0.9599 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8246 |
| Balanced Acc | 0.7873 |
| Macro-Sensitivity | 0.7873 |
| Macro-Specificity | 0.9647 |
| Macro-Precision | 0.7850 |
| AUC-PR (Macro) | 0.8461 |
| Cohen's Kappa | 0.7784 |
| MCC | 0.7789 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 10860.7 (0.09 ms/张) |
| End-to-End FPS | 607.8 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8794 | 0.9950 | 0.7709 | 0.8216 | 0.9977 | 0.9973 | 0.9094 | 199 |
| cancer | 0.9100 | 0.9667 | 0.9005 | 0.9052 | 0.9701 | 0.9879 | 0.9704 | 2645 |
| empty | 0.9957 | 0.9645 | 0.9252 | 0.9592 | 0.9980 | 0.9920 | 0.9787 | 3256 |
| normal | 0.7366 | 0.9691 | 0.8229 | 0.7774 | 0.9497 | 0.9540 | 0.8623 | 1735 |
| other | 0.6308 | 0.9237 | 0.6143 | 0.6224 | 0.9285 | 0.8927 | 0.6623 | 1717 |
| stroma | 0.5712 | 0.9690 | 0.6761 | 0.6192 | 0.9523 | 0.9355 | 0.6937 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=175 FP=52 TN=10383 FN=24
- **cancer**: TP=2407 FP=266 TN=7723 FN=238
- **empty**: TP=3242 FP=262 TN=7116 FN=14
- **normal**: TP=1278 FP=275 TN=8624 FN=457
- **other**: TP=1083 FP=680 TN=8237 FN=634
- **stroma**: TP=618 FP=296 TN=9256 FN=464

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 175 | 5 | 2 | 3 | 13 | 1 |
| cancer | 6 | 2407 | 1 | 151 | 53 | 27 |
| empty | 0 | 4 | 3242 | 0 | 10 | 0 |
| normal | 2 | 46 | 1 | 1278 | 264 | 144 |
| other | 43 | 175 | 258 | 34 | 1083 | 124 |
| stroma | 1 | 36 | 0 | 87 | 340 | 618 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7709    0.8794    0.8216       199
      cancer     0.9005    0.9100    0.9052      2645
       empty     0.9252    0.9957    0.9592      3256
      normal     0.8229    0.7366    0.7774      1735
       other     0.6143    0.6308    0.6224      1717
      stroma     0.6761    0.5712    0.6192      1082

    accuracy                         0.8278     10634
   macro avg     0.7850    0.7873    0.7842     10634
weighted avg     0.8239    0.8278    0.8246     10634

```