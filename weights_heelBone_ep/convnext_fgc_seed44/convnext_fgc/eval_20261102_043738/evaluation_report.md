# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_fgc_seed44/convnext_fgc/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8918 |
| Accuracy | 0.8820 |
| Macro-AUROC (OvR) | 0.9787 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8818 |
| Balanced Acc | 0.8967 |
| Macro-Sensitivity | 0.8967 |
| Macro-Specificity | 0.9363 |
| Macro-Precision | 0.8875 |
| AUC-PR (Macro) | 0.9692 |
| Cohen's Kappa | 0.8142 |
| MCC | 0.8145 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 935.1 (1.07 ms/张) |
| End-to-End FPS | 283.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8528 | 0.9141 | 0.8317 | 0.8421 | 0.9258 | 0.9679 | 0.9480 | 197 |
| normal | 0.8623 | 0.9117 | 0.8947 | 0.8782 | 0.8838 | 0.9694 | 0.9639 | 276 |
| sever | 0.9750 | 0.9831 | 0.9360 | 0.9551 | 0.9936 | 0.9988 | 0.9956 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=168 FP=34 TN=362 FN=29
- **normal**: TP=238 FP=28 TN=289 FN=38
- **sever**: TP=117 FP=8 TN=465 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 168 | 25 | 4 |
| normal | 34 | 238 | 4 |
| sever | 0 | 3 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8317    0.8528    0.8421       197
      normal     0.8947    0.8623    0.8782       276
       sever     0.9360    0.9750    0.9551       120

    accuracy                         0.8820       593
   macro avg     0.8875    0.8967    0.8918       593
weighted avg     0.8821    0.8820    0.8818       593

```