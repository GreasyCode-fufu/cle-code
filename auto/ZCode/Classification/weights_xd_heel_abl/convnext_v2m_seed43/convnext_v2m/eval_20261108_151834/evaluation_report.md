# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_abl/convnext_v2m_seed43/convnext_v2m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8882 |
| Accuracy | 0.8820 |
| Macro-AUROC (OvR) | 0.9789 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8815 |
| Balanced Acc | 0.9027 |
| Macro-Sensitivity | 0.9027 |
| Macro-Specificity | 0.9396 |
| Macro-Precision | 0.8783 |
| AUC-PR (Macro) | 0.9678 |
| Cohen's Kappa | 0.8161 |
| MCC | 0.8183 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 570.3 (1.75 ms/张) |
| End-to-End FPS | 233.6 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8832 | 0.9116 | 0.8325 | 0.8571 | 0.9401 | 0.9710 | 0.9500 | 197 |
| normal | 0.8333 | 0.9432 | 0.9274 | 0.8779 | 0.8667 | 0.9681 | 0.9640 | 276 |
| sever | 0.9917 | 0.9641 | 0.8750 | 0.9297 | 0.9978 | 0.9976 | 0.9895 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=174 FP=35 TN=361 FN=23
- **normal**: TP=230 FP=18 TN=299 FN=46
- **sever**: TP=119 FP=17 TN=456 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 174 | 18 | 5 |
| normal | 34 | 230 | 12 |
| sever | 1 | 0 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8325    0.8832    0.8571       197
      normal     0.9274    0.8333    0.8779       276
       sever     0.8750    0.9917    0.9297       120

    accuracy                         0.8820       593
   macro avg     0.8783    0.9027    0.8882       593
weighted avg     0.8853    0.8820    0.8815       593

```