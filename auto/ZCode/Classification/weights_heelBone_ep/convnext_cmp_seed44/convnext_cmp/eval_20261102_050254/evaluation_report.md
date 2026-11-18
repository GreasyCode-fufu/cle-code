# convnext_cmp 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_cmp_seed44/convnext_cmp/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8386 |
| Accuracy | 0.8347 |
| Macro-AUROC (OvR) | 0.9770 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8321 |
| Balanced Acc | 0.8744 |
| Macro-Sensitivity | 0.8744 |
| Macro-Specificity | 0.9214 |
| Macro-Precision | 0.8325 |
| AUC-PR (Macro) | 0.9670 |
| Cohen's Kappa | 0.7487 |
| MCC | 0.7632 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 834.2 (1.20 ms/张) |
| End-to-End FPS | 269.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9239 | 0.8687 | 0.7778 | 0.8445 | 0.9582 | 0.9660 | 0.9477 | 197 |
| normal | 0.6993 | 0.9779 | 0.9650 | 0.8109 | 0.7888 | 0.9663 | 0.9577 | 276 |
| sever | 1.0000 | 0.9175 | 0.7547 | 0.8602 | 1.0000 | 0.9988 | 0.9956 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=182 FP=52 TN=344 FN=15
- **normal**: TP=193 FP=7 TN=310 FN=83
- **sever**: TP=120 FP=39 TN=434 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 182 | 7 | 8 |
| normal | 52 | 193 | 31 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7778    0.9239    0.8445       197
      normal     0.9650    0.6993    0.8109       276
       sever     0.7547    1.0000    0.8602       120

    accuracy                         0.8347       593
   macro avg     0.8325    0.8744    0.8386       593
weighted avg     0.8603    0.8347    0.8321       593

```