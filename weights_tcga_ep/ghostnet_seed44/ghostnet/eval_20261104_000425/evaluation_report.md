# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/ghostnet_seed44/ghostnet/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7940 |
| Accuracy | 0.8300 |
| Macro-AUROC (OvR) | 0.9566 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8273 |
| Balanced Acc | 0.7939 |
| Macro-Sensitivity | 0.7939 |
| Macro-Specificity | 0.9650 |
| Macro-Precision | 0.7979 |
| AUC-PR (Macro) | 0.8597 |
| Cohen's Kappa | 0.7811 |
| MCC | 0.7820 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.59M |
| FLOPs | 309.17M |
| Params | 3.91M |
| Pure-Inference FPS | 4992.4 (0.20 ms/张) |
| End-to-End FPS | 571.5 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8995 | 0.9967 | 0.8404 | 0.8689 | 0.9981 | 0.9967 | 0.9513 | 199 |
| cancer | 0.9100 | 0.9583 | 0.8785 | 0.8940 | 0.9699 | 0.9872 | 0.9634 | 2645 |
| empty | 0.9948 | 0.9623 | 0.9210 | 0.9564 | 0.9976 | 0.9925 | 0.9784 | 3256 |
| normal | 0.7268 | 0.9836 | 0.8962 | 0.8027 | 0.9486 | 0.9734 | 0.9031 | 1735 |
| other | 0.6407 | 0.9334 | 0.6494 | 0.6450 | 0.9310 | 0.8872 | 0.7057 | 1717 |
| stroma | 0.5915 | 0.9557 | 0.6021 | 0.5967 | 0.9538 | 0.9026 | 0.6564 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=179 FP=34 TN=10401 FN=20
- **cancer**: TP=2407 FP=333 TN=7656 FN=238
- **empty**: TP=3239 FP=278 TN=7100 FN=17
- **normal**: TP=1261 FP=146 TN=8753 FN=474
- **other**: TP=1100 FP=594 TN=8323 FN=617
- **stroma**: TP=640 FP=423 TN=9129 FN=442

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 179 | 4 | 2 | 4 | 9 | 1 |
| cancer | 3 | 2407 | 1 | 74 | 102 | 58 |
| empty | 0 | 5 | 3239 | 0 | 12 | 0 |
| normal | 5 | 111 | 4 | 1261 | 148 | 206 |
| other | 23 | 144 | 271 | 21 | 1100 | 158 |
| stroma | 3 | 69 | 0 | 47 | 323 | 640 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8404    0.8995    0.8689       199
      cancer     0.8785    0.9100    0.8940      2645
       empty     0.9210    0.9948    0.9564      3256
      normal     0.8962    0.7268    0.8027      1735
       other     0.6494    0.6407    0.6450      1717
      stroma     0.6021    0.5915    0.5967      1082

    accuracy                         0.8300     10634
   macro avg     0.7979    0.7939    0.7940     10634
weighted avg     0.8285    0.8300    0.8273     10634

```