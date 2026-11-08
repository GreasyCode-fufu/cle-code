# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_abl/convnext_v2m_seed42/convnext_v2m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8656 |
| Accuracy | 0.8634 |
| Macro-AUROC (OvR) | 0.9806 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8628 |
| Balanced Acc | 0.8841 |
| Macro-Sensitivity | 0.8841 |
| Macro-Specificity | 0.9305 |
| Macro-Precision | 0.8546 |
| AUC-PR (Macro) | 0.9714 |
| Cohen's Kappa | 0.7875 |
| MCC | 0.7898 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1650.3 (0.61 ms/张) |
| End-to-End FPS | 332.2 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8274 | 0.9318 | 0.8579 | 0.8424 | 0.9156 | 0.9708 | 0.9500 | 197 |
| normal | 0.8333 | 0.9211 | 0.9020 | 0.8663 | 0.8639 | 0.9725 | 0.9682 | 276 |
| sever | 0.9917 | 0.9387 | 0.8041 | 0.8881 | 0.9978 | 0.9984 | 0.9961 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=163 FP=27 TN=369 FN=34
- **normal**: TP=230 FP=25 TN=292 FN=46
- **sever**: TP=119 FP=29 TN=444 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 163 | 24 | 10 |
| normal | 27 | 230 | 19 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8579    0.8274    0.8424       197
      normal     0.9020    0.8333    0.8663       276
       sever     0.8041    0.9917    0.8881       120

    accuracy                         0.8634       593
   macro avg     0.8546    0.8841    0.8656       593
weighted avg     0.8675    0.8634    0.8628       593

```