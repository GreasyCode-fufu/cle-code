# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_fgc_seed44/convnext_fgc/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8260 |
| Accuracy | 0.8475 |
| Macro-AUROC (OvR) | 0.9469 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8458 |
| Balanced Acc | 0.8259 |
| Macro-Sensitivity | 0.8259 |
| Macro-Specificity | 0.9098 |
| Macro-Precision | 0.8314 |
| AUC-PR (Macro) | 0.9060 |
| Cohen's Kappa | 0.7367 |
| MCC | 0.7385 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 477.7 (2.09 ms/张) |
| End-to-End FPS | 140.3 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9091 | 0.8269 | 0.8696 | 0.8889 | 0.8776 | 0.9235 | 0.9487 | 66 |
| malignant | 0.7188 | 0.9535 | 0.8519 | 0.7797 | 0.9011 | 0.9477 | 0.8679 | 32 |
| normal | 0.8500 | 0.9490 | 0.7727 | 0.8095 | 0.9688 | 0.9694 | 0.9013 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=9 TN=43 FN=6
- **malignant**: TP=23 FP=4 TN=82 FN=9
- **normal**: TP=17 FP=5 TN=93 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 4 | 2 |
| malignant | 6 | 23 | 3 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8696    0.9091    0.8889        66
   malignant     0.8519    0.7188    0.7797        32
      normal     0.7727    0.8500    0.8095        20

    accuracy                         0.8475       118
   macro avg     0.8314    0.8259    0.8260       118
weighted avg     0.8483    0.8475    0.8458       118

```