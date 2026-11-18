# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/resnet50_seed44/resnet50/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8295 |
| Accuracy | 0.8448 |
| Macro-AUROC (OvR) | 0.9360 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8463 |
| Balanced Acc | 0.8568 |
| Macro-Sensitivity | 0.8568 |
| Macro-Specificity | 0.9199 |
| Macro-Precision | 0.8160 |
| AUC-PR (Macro) | 0.8875 |
| Cohen's Kappa | 0.7428 |
| MCC | 0.7466 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 1451.1 (0.69 ms/张) |
| End-to-End FPS | 172.6 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8462 | 0.9020 | 0.9167 | 0.8800 | 0.8214 | 0.9376 | 0.9323 | 65 |
| malignant | 0.7742 | 0.9412 | 0.8276 | 0.8000 | 0.9195 | 0.9294 | 0.8478 | 31 |
| normal | 0.9500 | 0.9167 | 0.7037 | 0.8085 | 0.9888 | 0.9411 | 0.8825 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=55 FP=5 TN=46 FN=10
- **malignant**: TP=24 FP=5 TN=80 FN=7
- **normal**: TP=19 FP=8 TN=88 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 55 | 5 | 5 |
| malignant | 4 | 24 | 3 |
| normal | 1 | 0 | 19 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9167    0.8462    0.8800        65
   malignant     0.8276    0.7742    0.8000        31
      normal     0.7037    0.9500    0.8085        20

    accuracy                         0.8448       116
   macro avg     0.8160    0.8568    0.8295       116
weighted avg     0.8561    0.8448    0.8463       116

```