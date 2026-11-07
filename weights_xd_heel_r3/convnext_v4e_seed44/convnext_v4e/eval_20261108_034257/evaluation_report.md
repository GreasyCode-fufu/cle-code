# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r3/convnext_v4e_seed44/convnext_v4e/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8644 |
| Accuracy | 0.8516 |
| Macro-AUROC (OvR) | 0.9755 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8505 |
| Balanced Acc | 0.8825 |
| Macro-Sensitivity | 0.8825 |
| Macro-Specificity | 0.9242 |
| Macro-Precision | 0.8572 |
| AUC-PR (Macro) | 0.9651 |
| Cohen's Kappa | 0.7703 |
| MCC | 0.7769 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 576.4 (1.73 ms/张) |
| End-to-End FPS | 234.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8985 | 0.8611 | 0.7629 | 0.8252 | 0.9446 | 0.9625 | 0.9394 | 197 |
| normal | 0.7572 | 0.9432 | 0.9207 | 0.8310 | 0.8169 | 0.9648 | 0.9583 | 276 |
| sever | 0.9917 | 0.9683 | 0.8881 | 0.9370 | 0.9978 | 0.9993 | 0.9976 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=177 FP=55 TN=341 FN=20
- **normal**: TP=209 FP=18 TN=299 FN=67
- **sever**: TP=119 FP=15 TN=458 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 177 | 17 | 3 |
| normal | 55 | 209 | 12 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7629    0.8985    0.8252       197
      normal     0.9207    0.7572    0.8310       276
       sever     0.8881    0.9917    0.9370       120

    accuracy                         0.8516       593
   macro avg     0.8572    0.8825    0.8644       593
weighted avg     0.8617    0.8516    0.8505       593

```