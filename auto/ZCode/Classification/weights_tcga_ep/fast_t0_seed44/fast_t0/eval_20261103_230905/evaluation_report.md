# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/fast_t0_seed44/fast_t0/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7868 |
| Accuracy | 0.8256 |
| Macro-AUROC (OvR) | 0.9607 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8235 |
| Balanced Acc | 0.7846 |
| Macro-Sensitivity | 0.7846 |
| Macro-Specificity | 0.9642 |
| Macro-Precision | 0.7910 |
| AUC-PR (Macro) | 0.8608 |
| Cohen's Kappa | 0.7756 |
| MCC | 0.7762 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 13703.0 (0.07 ms/张) |
| End-to-End FPS | 597.5 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8543 | 0.9967 | 0.8333 | 0.8437 | 0.9972 | 0.9981 | 0.9310 | 199 |
| cancer | 0.9009 | 0.9655 | 0.8962 | 0.8986 | 0.9671 | 0.9896 | 0.9727 | 2645 |
| empty | 0.9923 | 0.9615 | 0.9192 | 0.9544 | 0.9965 | 0.9913 | 0.9778 | 3256 |
| normal | 0.7568 | 0.9800 | 0.8806 | 0.8140 | 0.9538 | 0.9785 | 0.9095 | 1735 |
| other | 0.5981 | 0.9258 | 0.6081 | 0.6031 | 0.9229 | 0.8736 | 0.6844 | 1717 |
| stroma | 0.6054 | 0.9559 | 0.6087 | 0.6070 | 0.9553 | 0.9332 | 0.6894 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=170 FP=34 TN=10401 FN=29
- **cancer**: TP=2383 FP=276 TN=7713 FN=262
- **empty**: TP=3231 FP=284 TN=7094 FN=25
- **normal**: TP=1313 FP=178 TN=8721 FN=422
- **other**: TP=1027 FP=662 TN=8255 FN=690
- **stroma**: TP=655 FP=421 TN=9131 FN=427

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 170 | 8 | 2 | 3 | 12 | 4 |
| cancer | 3 | 2383 | 1 | 79 | 124 | 55 |
| empty | 0 | 2 | 3231 | 0 | 23 | 0 |
| normal | 0 | 71 | 0 | 1313 | 207 | 144 |
| other | 30 | 126 | 281 | 35 | 1027 | 218 |
| stroma | 1 | 69 | 0 | 61 | 296 | 655 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8333    0.8543    0.8437       199
      cancer     0.8962    0.9009    0.8986      2645
       empty     0.9192    0.9923    0.9544      3256
      normal     0.8806    0.7568    0.8140      1735
       other     0.6081    0.5981    0.6031      1717
      stroma     0.6087    0.6054    0.6070      1082

    accuracy                         0.8256     10634
   macro avg     0.7910    0.7846    0.7868     10634
weighted avg     0.8238    0.8256    0.8235     10634

```