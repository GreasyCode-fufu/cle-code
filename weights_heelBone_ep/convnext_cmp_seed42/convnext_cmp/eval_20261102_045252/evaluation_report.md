# convnext_cmp 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_cmp_seed42/convnext_cmp/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8831 |
| Accuracy | 0.8735 |
| Macro-AUROC (OvR) | 0.9793 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8733 |
| Balanced Acc | 0.8991 |
| Macro-Sensitivity | 0.8991 |
| Macro-Specificity | 0.9363 |
| Macro-Precision | 0.8760 |
| AUC-PR (Macro) | 0.9699 |
| Cohen's Kappa | 0.8039 |
| MCC | 0.8094 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 992.6 (1.01 ms/张) |
| End-to-End FPS | 286.2 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9086 | 0.8763 | 0.7851 | 0.8424 | 0.9507 | 0.9658 | 0.9457 | 197 |
| normal | 0.7971 | 0.9621 | 0.9483 | 0.8661 | 0.8449 | 0.9735 | 0.9687 | 276 |
| sever | 0.9917 | 0.9704 | 0.8947 | 0.9407 | 0.9978 | 0.9986 | 0.9954 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=179 FP=49 TN=347 FN=18
- **normal**: TP=220 FP=12 TN=305 FN=56
- **sever**: TP=119 FP=14 TN=459 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 179 | 12 | 6 |
| normal | 48 | 220 | 8 |
| sever | 1 | 0 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7851    0.9086    0.8424       197
      normal     0.9483    0.7971    0.8661       276
       sever     0.8947    0.9917    0.9407       120

    accuracy                         0.8735       593
   macro avg     0.8760    0.8991    0.8831       593
weighted avg     0.8832    0.8735    0.8733       593

```