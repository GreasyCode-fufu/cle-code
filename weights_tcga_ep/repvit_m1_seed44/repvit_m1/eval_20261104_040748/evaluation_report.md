# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/repvit_m1_seed44/repvit_m1/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7734 |
| Accuracy | 0.8186 |
| Macro-AUROC (OvR) | 0.9498 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8180 |
| Balanced Acc | 0.7915 |
| Macro-Sensitivity | 0.7915 |
| Macro-Specificity | 0.9633 |
| Macro-Precision | 0.7704 |
| AUC-PR (Macro) | 0.8353 |
| Cohen's Kappa | 0.7677 |
| MCC | 0.7697 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 4920.1 (0.20 ms/张) |
| End-to-End FPS | 589.5 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8844 | 0.9928 | 0.7012 | 0.7822 | 0.9978 | 0.9946 | 0.8960 | 199 |
| cancer | 0.8764 | 0.9678 | 0.9002 | 0.8881 | 0.9594 | 0.9856 | 0.9616 | 2645 |
| empty | 0.9948 | 0.9622 | 0.9207 | 0.9563 | 0.9976 | 0.9931 | 0.9820 | 3256 |
| normal | 0.6640 | 0.9828 | 0.8828 | 0.7579 | 0.9375 | 0.9055 | 0.8271 | 1735 |
| other | 0.6005 | 0.9462 | 0.6823 | 0.6388 | 0.9248 | 0.8909 | 0.7028 | 1717 |
| stroma | 0.7292 | 0.9283 | 0.5353 | 0.6174 | 0.9680 | 0.9294 | 0.6422 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=176 FP=75 TN=10360 FN=23
- **cancer**: TP=2318 FP=257 TN=7732 FN=327
- **empty**: TP=3239 FP=279 TN=7099 FN=17
- **normal**: TP=1152 FP=153 TN=8746 FN=583
- **other**: TP=1031 FP=480 TN=8437 FN=686
- **stroma**: TP=789 FP=685 TN=8867 FN=293

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 176 | 2 | 3 | 2 | 13 | 3 |
| cancer | 10 | 2318 | 1 | 66 | 115 | 135 |
| empty | 0 | 2 | 3239 | 0 | 15 | 0 |
| normal | 6 | 115 | 0 | 1152 | 150 | 312 |
| other | 56 | 99 | 275 | 21 | 1031 | 235 |
| stroma | 3 | 39 | 0 | 64 | 187 | 789 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7012    0.8844    0.7822       199
      cancer     0.9002    0.8764    0.8881      2645
       empty     0.9207    0.9948    0.9563      3256
      normal     0.8828    0.6640    0.7579      1735
       other     0.6823    0.6005    0.6388      1717
      stroma     0.5353    0.7292    0.6174      1082

    accuracy                         0.8186     10634
   macro avg     0.7704    0.7915    0.7734     10634
weighted avg     0.8276    0.8186    0.8180     10634

```