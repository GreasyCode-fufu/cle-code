# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/swin_tiny_vanilla_patch4_window7_224_seed44/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8441 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9436 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8719 |
| Balanced Acc | 0.8547 |
| Macro-Sensitivity | 0.8547 |
| Macro-Specificity | 0.9334 |
| Macro-Precision | 0.8363 |
| AUC-PR (Macro) | 0.8822 |
| Cohen's Kappa | 0.7817 |
| MCC | 0.7825 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 1313.5 (0.76 ms/张) |
| End-to-End FPS | 170.6 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9077 | 0.9216 | 0.9365 | 0.9219 | 0.8868 | 0.9575 | 0.9583 | 65 |
| malignant | 0.8065 | 0.9412 | 0.8333 | 0.8197 | 0.9302 | 0.9306 | 0.8960 | 31 |
| normal | 0.8500 | 0.9375 | 0.7391 | 0.7907 | 0.9677 | 0.9427 | 0.7921 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=4 TN=47 FN=6
- **malignant**: TP=25 FP=5 TN=80 FN=6
- **normal**: TP=17 FP=6 TN=90 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 5 | 1 |
| malignant | 1 | 25 | 5 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9365    0.9077    0.9219        65
   malignant     0.8333    0.8065    0.8197        31
      normal     0.7391    0.8500    0.7907        20

    accuracy                         0.8707       116
   macro avg     0.8363    0.8547    0.8441       116
weighted avg     0.8749    0.8707    0.8719       116

```