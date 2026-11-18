# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mobilenet_v3_large_seed42/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8174 |
| Accuracy | 0.7943 |
| Macro-AUROC (OvR) | 0.9481 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7946 |
| Balanced Acc | 0.8316 |
| Macro-Sensitivity | 0.8316 |
| Macro-Specificity | 0.8939 |
| Macro-Precision | 0.8251 |
| AUC-PR (Macro) | 0.9221 |
| Cohen's Kappa | 0.6827 |
| MCC | 0.6971 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 3271.2 (0.31 ms/张) |
| End-to-End FPS | 241.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8782 | 0.7702 | 0.6553 | 0.7505 | 0.9271 | 0.9223 | 0.8680 | 197 |
| normal | 0.6667 | 0.9306 | 0.8932 | 0.7635 | 0.7623 | 0.9256 | 0.9089 | 276 |
| sever | 0.9500 | 0.9810 | 0.9268 | 0.9383 | 0.9872 | 0.9964 | 0.9894 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=173 FP=91 TN=305 FN=24
- **normal**: TP=184 FP=22 TN=295 FN=92
- **sever**: TP=114 FP=9 TN=464 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 173 | 20 | 4 |
| normal | 87 | 184 | 5 |
| sever | 4 | 2 | 114 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6553    0.8782    0.7505       197
      normal     0.8932    0.6667    0.7635       276
       sever     0.9268    0.9500    0.9383       120

    accuracy                         0.7943       593
   macro avg     0.8251    0.8316    0.8174       593
weighted avg     0.8210    0.7943    0.7946       593

```