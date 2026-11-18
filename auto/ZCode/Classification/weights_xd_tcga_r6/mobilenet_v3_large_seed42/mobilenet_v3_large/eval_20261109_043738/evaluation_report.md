# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mobilenet_v3_large_seed42/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7923 |
| Accuracy | 0.8315 |
| Macro-AUROC (OvR) | 0.9588 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8284 |
| Balanced Acc | 0.7872 |
| Macro-Sensitivity | 0.7872 |
| Macro-Specificity | 0.9650 |
| Macro-Precision | 0.8012 |
| AUC-PR (Macro) | 0.8650 |
| Cohen's Kappa | 0.7828 |
| MCC | 0.7835 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.93M |
| Params | 4.21M |
| Pure-Inference FPS | 6059.2 (0.17 ms/张) |
| End-to-End FPS | 230.7 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8241 | 0.9961 | 0.8000 | 0.8119 | 0.9966 | 0.9934 | 0.8981 | 199 |
| cancer | 0.8881 | 0.9494 | 0.8533 | 0.8703 | 0.9624 | 0.9803 | 0.9358 | 2645 |
| empty | 0.9917 | 0.9629 | 0.9218 | 0.9555 | 0.9962 | 0.9948 | 0.9877 | 3256 |
| normal | 0.7470 | 0.9765 | 0.8611 | 0.8000 | 0.9519 | 0.9339 | 0.8673 | 1735 |
| other | 0.6733 | 0.9322 | 0.6564 | 0.6647 | 0.9368 | 0.9180 | 0.7661 | 1717 |
| stroma | 0.5989 | 0.9729 | 0.7144 | 0.6516 | 0.9554 | 0.9326 | 0.7350 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=164 FP=41 TN=10394 FN=35
- **cancer**: TP=2349 FP=404 TN=7585 FN=296
- **empty**: TP=3229 FP=274 TN=7104 FN=27
- **normal**: TP=1296 FP=209 TN=8690 FN=439
- **other**: TP=1156 FP=605 TN=8312 FN=561
- **stroma**: TP=648 FP=259 TN=9293 FN=434

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 164 | 3 | 2 | 2 | 27 | 1 |
| cancer | 5 | 2349 | 0 | 98 | 155 | 38 |
| empty | 0 | 5 | 3229 | 0 | 22 | 0 |
| normal | 4 | 163 | 2 | 1296 | 147 | 123 |
| other | 30 | 136 | 270 | 28 | 1156 | 97 |
| stroma | 2 | 97 | 0 | 81 | 254 | 648 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8000    0.8241    0.8119       199
      cancer     0.8533    0.8881    0.8703      2645
       empty     0.9218    0.9917    0.9555      3256
      normal     0.8611    0.7470    0.8000      1735
       other     0.6564    0.6733    0.6647      1717
      stroma     0.7144    0.5989    0.6516      1082

    accuracy                         0.8315     10634
   macro avg     0.8012    0.7872    0.7923     10634
weighted avg     0.8286    0.8315    0.8284     10634

```