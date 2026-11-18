# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_dce4m_seed44/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8709 |
| Accuracy | 0.8600 |
| Macro-AUROC (OvR) | 0.9620 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8594 |
| Balanced Acc | 0.8756 |
| Macro-Sensitivity | 0.8756 |
| Macro-Specificity | 0.9233 |
| Macro-Precision | 0.8669 |
| AUC-PR (Macro) | 0.9446 |
| Cohen's Kappa | 0.7792 |
| MCC | 0.7794 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3959.5 (0.25 ms/张) |
| End-to-End FPS | 20.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8122 | 0.9192 | 0.8333 | 0.8226 | 0.9077 | 0.9450 | 0.9162 | 197 |
| normal | 0.8478 | 0.8738 | 0.8540 | 0.8509 | 0.8683 | 0.9429 | 0.9251 | 276 |
| sever | 0.9667 | 0.9767 | 0.9134 | 0.9393 | 0.9914 | 0.9980 | 0.9924 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=160 FP=32 TN=364 FN=37
- **normal**: TP=234 FP=40 TN=277 FN=42
- **sever**: TP=116 FP=11 TN=462 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 160 | 36 | 1 |
| normal | 32 | 234 | 10 |
| sever | 0 | 4 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8333    0.8122    0.8226       197
      normal     0.8540    0.8478    0.8509       276
       sever     0.9134    0.9667    0.9393       120

    accuracy                         0.8600       593
   macro avg     0.8669    0.8756    0.8709       593
weighted avg     0.8592    0.8600    0.8594       593

```