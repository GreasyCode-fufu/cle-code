# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_abl/convnext_v2m_seed44/convnext_v2m/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8083 |
| Accuracy | 0.8426 |
| Macro-AUROC (OvR) | 0.9589 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8370 |
| Balanced Acc | 0.8088 |
| Macro-Sensitivity | 0.8088 |
| Macro-Specificity | 0.9671 |
| Macro-Precision | 0.8138 |
| AUC-PR (Macro) | 0.8628 |
| Cohen's Kappa | 0.7969 |
| MCC | 0.7985 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 605.2 (1.65 ms/张) |
| End-to-End FPS | 316.4 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8744 | 0.9961 | 0.8093 | 0.8406 | 0.9976 | 0.9979 | 0.9285 | 199 |
| cancer | 0.9391 | 0.9419 | 0.8426 | 0.8883 | 0.9791 | 0.9865 | 0.9657 | 2645 |
| empty | 0.9966 | 0.9625 | 0.9214 | 0.9575 | 0.9985 | 0.9733 | 0.8876 | 3256 |
| normal | 0.7418 | 0.9826 | 0.8925 | 0.8102 | 0.9513 | 0.9645 | 0.8917 | 1735 |
| other | 0.5708 | 0.9505 | 0.6897 | 0.6246 | 0.9200 | 0.8670 | 0.7010 | 1717 |
| stroma | 0.7301 | 0.9690 | 0.7274 | 0.7288 | 0.9694 | 0.9639 | 0.8024 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=174 FP=41 TN=10394 FN=25
- **cancer**: TP=2484 FP=464 TN=7525 FN=161
- **empty**: TP=3245 FP=277 TN=7101 FN=11
- **normal**: TP=1287 FP=155 TN=8744 FN=448
- **other**: TP=980 FP=441 TN=8476 FN=737
- **stroma**: TP=790 FP=296 TN=9256 FN=292

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 174 | 10 | 2 | 3 | 8 | 2 |
| cancer | 4 | 2484 | 1 | 59 | 68 | 29 |
| empty | 0 | 4 | 3245 | 1 | 6 | 0 |
| normal | 1 | 145 | 0 | 1287 | 215 | 87 |
| other | 33 | 233 | 274 | 19 | 980 | 178 |
| stroma | 3 | 72 | 0 | 73 | 144 | 790 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8093    0.8744    0.8406       199
      cancer     0.8426    0.9391    0.8883      2645
       empty     0.9214    0.9966    0.9575      3256
      normal     0.8925    0.7418    0.8102      1735
       other     0.6897    0.5708    0.6246      1717
      stroma     0.7274    0.7301    0.7288      1082

    accuracy                         0.8426     10634
   macro avg     0.8138    0.8088    0.8083     10634
weighted avg     0.8378    0.8426    0.8370     10634

```