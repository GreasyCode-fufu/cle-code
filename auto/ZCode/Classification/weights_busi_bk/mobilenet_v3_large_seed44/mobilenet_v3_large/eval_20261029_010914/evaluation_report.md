# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/mobilenet_v3_large_seed44/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8098 |
| Accuracy | 0.8305 |
| Macro-AUROC (OvR) | 0.9417 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8298 |
| Balanced Acc | 0.8158 |
| Macro-Sensitivity | 0.8158 |
| Macro-Specificity | 0.9025 |
| Macro-Precision | 0.8087 |
| AUC-PR (Macro) | 0.9031 |
| Cohen's Kappa | 0.7103 |
| MCC | 0.7114 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 1729.4 (0.58 ms/张) |
| End-to-End FPS | 183.5 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.8269 | 0.8657 | 0.8722 | 0.8431 | 0.9280 | 0.9420 | 66 |
| malignant | 0.7188 | 0.9419 | 0.8214 | 0.7667 | 0.9000 | 0.9393 | 0.8734 | 32 |
| normal | 0.8500 | 0.9388 | 0.7391 | 0.7907 | 0.9684 | 0.9577 | 0.8940 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=9 TN=43 FN=8
- **malignant**: TP=23 FP=5 TN=81 FN=9
- **normal**: TP=17 FP=6 TN=92 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 5 | 3 |
| malignant | 6 | 23 | 3 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8657    0.8788    0.8722        66
   malignant     0.8214    0.7188    0.7667        32
      normal     0.7391    0.8500    0.7907        20

    accuracy                         0.8305       118
   macro avg     0.8087    0.8158    0.8098       118
weighted avg     0.8322    0.8305    0.8298       118

```