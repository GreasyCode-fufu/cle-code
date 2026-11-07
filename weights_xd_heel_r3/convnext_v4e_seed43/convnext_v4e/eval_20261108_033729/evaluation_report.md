# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r3/convnext_v4e_seed43/convnext_v4e/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8836 |
| Accuracy | 0.8836 |
| Macro-AUROC (OvR) | 0.9772 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8825 |
| Balanced Acc | 0.8948 |
| Macro-Sensitivity | 0.8948 |
| Macro-Specificity | 0.9382 |
| Macro-Precision | 0.8820 |
| AUC-PR (Macro) | 0.9677 |
| Cohen's Kappa | 0.8169 |
| MCC | 0.8200 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 454.1 (2.20 ms/张) |
| End-to-End FPS | 210.1 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7868 | 0.9747 | 0.9394 | 0.8564 | 0.9019 | 0.9677 | 0.9499 | 197 |
| normal | 0.9058 | 0.8927 | 0.8803 | 0.8929 | 0.9159 | 0.9644 | 0.9550 | 276 |
| sever | 0.9917 | 0.9471 | 0.8264 | 0.9015 | 0.9978 | 0.9995 | 0.9982 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=155 FP=10 TN=386 FN=42
- **normal**: TP=250 FP=34 TN=283 FN=26
- **sever**: TP=119 FP=25 TN=448 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 155 | 33 | 9 |
| normal | 10 | 250 | 16 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.9394    0.7868    0.8564       197
      normal     0.8803    0.9058    0.8929       276
       sever     0.8264    0.9917    0.9015       120

    accuracy                         0.8836       593
   macro avg     0.8820    0.8948    0.8836       593
weighted avg     0.8890    0.8836    0.8825       593

```