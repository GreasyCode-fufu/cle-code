# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/jumbo_small_seed44/jumbo_small/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8180 |
| Accuracy | 0.8448 |
| Macro-AUROC (OvR) | 0.9076 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8438 |
| Balanced Acc | 0.8165 |
| Macro-Sensitivity | 0.8165 |
| Macro-Specificity | 0.9107 |
| Macro-Precision | 0.8213 |
| AUC-PR (Macro) | 0.8241 |
| Cohen's Kappa | 0.7328 |
| MCC | 0.7334 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 1135.2 (0.88 ms/张) |
| End-to-End FPS | 160.9 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9077 | 0.8431 | 0.8806 | 0.8939 | 0.8776 | 0.9216 | 0.9201 | 65 |
| malignant | 0.7419 | 0.9412 | 0.8214 | 0.7797 | 0.9091 | 0.8600 | 0.8270 | 31 |
| normal | 0.8000 | 0.9479 | 0.7619 | 0.7805 | 0.9579 | 0.9411 | 0.7252 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=8 TN=43 FN=6
- **malignant**: TP=23 FP=5 TN=80 FN=8
- **normal**: TP=16 FP=5 TN=91 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 5 | 1 |
| malignant | 4 | 23 | 4 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8806    0.9077    0.8939        65
   malignant     0.8214    0.7419    0.7797        31
      normal     0.7619    0.8000    0.7805        20

    accuracy                         0.8448       116
   macro avg     0.8213    0.8165    0.8180       116
weighted avg     0.8443    0.8448    0.8438       116

```