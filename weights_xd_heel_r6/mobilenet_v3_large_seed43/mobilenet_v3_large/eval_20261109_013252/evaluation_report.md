# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mobilenet_v3_large_seed43/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8253 |
| Accuracy | 0.8111 |
| Macro-AUROC (OvR) | 0.9528 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8088 |
| Balanced Acc | 0.8500 |
| Macro-Sensitivity | 0.8500 |
| Macro-Specificity | 0.9045 |
| Macro-Precision | 0.8199 |
| AUC-PR (Macro) | 0.9266 |
| Cohen's Kappa | 0.7100 |
| MCC | 0.7209 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 3792.7 (0.26 ms/张) |
| End-to-End FPS | 324.9 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8782 | 0.8232 | 0.7119 | 0.7864 | 0.9314 | 0.9310 | 0.8851 | 197 |
| normal | 0.6884 | 0.9369 | 0.9048 | 0.7819 | 0.7755 | 0.9313 | 0.9054 | 276 |
| sever | 0.9833 | 0.9535 | 0.8429 | 0.9077 | 0.9956 | 0.9961 | 0.9892 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=173 FP=70 TN=326 FN=24
- **normal**: TP=190 FP=20 TN=297 FN=86
- **sever**: TP=118 FP=22 TN=451 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 173 | 18 | 6 |
| normal | 70 | 190 | 16 |
| sever | 0 | 2 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7119    0.8782    0.7864       197
      normal     0.9048    0.6884    0.7819       276
       sever     0.8429    0.9833    0.9077       120

    accuracy                         0.8111       593
   macro avg     0.8199    0.8500    0.8253       593
weighted avg     0.8282    0.8111    0.8088       593

```