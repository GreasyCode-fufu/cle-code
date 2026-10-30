# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/convnext_small_seed42/convnext_small/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8034 |
| Accuracy | 0.8305 |
| Macro-AUROC (OvR) | 0.9485 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8308 |
| Balanced Acc | 0.8221 |
| Macro-Sensitivity | 0.8221 |
| Macro-Specificity | 0.9090 |
| Macro-Precision | 0.8017 |
| AUC-PR (Macro) | 0.8810 |
| Cohen's Kappa | 0.7144 |
| MCC | 0.7181 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.41M |
| Pure-Inference FPS | 911.2 (1.10 ms/张) |
| End-to-End FPS | 168.0 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.8654 | 0.8923 | 0.8855 | 0.8491 | 0.9245 | 0.9537 | 66 |
| malignant | 0.6875 | 0.9535 | 0.8462 | 0.7586 | 0.8913 | 0.9440 | 0.7562 | 32 |
| normal | 0.9000 | 0.9082 | 0.6667 | 0.7660 | 0.9780 | 0.9770 | 0.9332 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=7 TN=45 FN=8
- **malignant**: TP=22 FP=4 TN=82 FN=10
- **normal**: TP=18 FP=9 TN=89 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 4 | 4 |
| malignant | 5 | 22 | 5 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8923    0.8788    0.8855        66
   malignant     0.8462    0.6875    0.7586        32
      normal     0.6667    0.9000    0.7660        20

    accuracy                         0.8305       118
   macro avg     0.8017    0.8221    0.8034       118
weighted avg     0.8415    0.8305    0.8308       118

```