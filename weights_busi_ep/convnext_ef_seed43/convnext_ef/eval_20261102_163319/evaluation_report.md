# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_ef_seed43/convnext_ef/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8007 |
| Accuracy | 0.8220 |
| Macro-AUROC (OvR) | 0.9414 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8192 |
| Balanced Acc | 0.8001 |
| Macro-Sensitivity | 0.8001 |
| Macro-Specificity | 0.8910 |
| Macro-Precision | 0.8150 |
| AUC-PR (Macro) | 0.8999 |
| Cohen's Kappa | 0.6906 |
| MCC | 0.6951 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 322.4 (3.10 ms/张) |
| End-to-End FPS | 123.4 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8939 | 0.7692 | 0.8310 | 0.8613 | 0.8511 | 0.9273 | 0.9426 | 66 |
| malignant | 0.6562 | 0.9651 | 0.8750 | 0.7500 | 0.8830 | 0.9244 | 0.8829 | 32 |
| normal | 0.8500 | 0.9388 | 0.7391 | 0.7907 | 0.9684 | 0.9724 | 0.8741 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=12 TN=40 FN=7
- **malignant**: TP=21 FP=3 TN=83 FN=11
- **normal**: TP=17 FP=6 TN=92 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 3 | 4 |
| malignant | 9 | 21 | 2 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8310    0.8939    0.8613        66
   malignant     0.8750    0.6562    0.7500        32
      normal     0.7391    0.8500    0.7907        20

    accuracy                         0.8220       118
   macro avg     0.8150    0.8001    0.8007       118
weighted avg     0.8274    0.8220    0.8192       118

```