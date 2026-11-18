# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mobilenet_v3_large_seed44/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8185 |
| Accuracy | 0.8111 |
| Macro-AUROC (OvR) | 0.9518 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8089 |
| Balanced Acc | 0.8466 |
| Macro-Sensitivity | 0.8466 |
| Macro-Specificity | 0.9053 |
| Macro-Precision | 0.8067 |
| AUC-PR (Macro) | 0.9298 |
| Cohen's Kappa | 0.7101 |
| MCC | 0.7172 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 2308.5 (0.43 ms/张) |
| End-to-End FPS | 177.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8426 | 0.8687 | 0.7615 | 0.8000 | 0.9173 | 0.9375 | 0.8958 | 197 |
| normal | 0.7138 | 0.9211 | 0.8874 | 0.7912 | 0.7871 | 0.9232 | 0.9114 | 276 |
| sever | 0.9833 | 0.9260 | 0.7712 | 0.8645 | 0.9955 | 0.9946 | 0.9823 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=166 FP=52 TN=344 FN=31
- **normal**: TP=197 FP=25 TN=292 FN=79
- **sever**: TP=118 FP=35 TN=438 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 166 | 23 | 8 |
| normal | 52 | 197 | 27 |
| sever | 0 | 2 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7615    0.8426    0.8000       197
      normal     0.8874    0.7138    0.7912       276
       sever     0.7712    0.9833    0.8645       120

    accuracy                         0.8111       593
   macro avg     0.8067    0.8466    0.8185       593
weighted avg     0.8221    0.8111    0.8089       593

```