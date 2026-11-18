# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce4m_seed44/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8420 |
| Accuracy | 0.8280 |
| Macro-AUROC (OvR) | 0.9308 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8269 |
| Balanced Acc | 0.8443 |
| Macro-Sensitivity | 0.8443 |
| Macro-Specificity | 0.9044 |
| Macro-Precision | 0.8406 |
| AUC-PR (Macro) | 0.8959 |
| Cohen's Kappa | 0.7277 |
| MCC | 0.7280 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3684.2 (0.27 ms/张) |
| End-to-End FPS | 20.0 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7411 | 0.9015 | 0.7892 | 0.7644 | 0.8750 | 0.8981 | 0.8393 | 197 |
| normal | 0.8333 | 0.8328 | 0.8127 | 0.8229 | 0.8516 | 0.8989 | 0.8649 | 276 |
| sever | 0.9583 | 0.9789 | 0.9200 | 0.9388 | 0.9893 | 0.9953 | 0.9836 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=146 FP=39 TN=357 FN=51
- **normal**: TP=230 FP=53 TN=264 FN=46
- **sever**: TP=115 FP=10 TN=463 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 146 | 49 | 2 |
| normal | 38 | 230 | 8 |
| sever | 1 | 4 | 115 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7892    0.7411    0.7644       197
      normal     0.8127    0.8333    0.8229       276
       sever     0.9200    0.9583    0.9388       120

    accuracy                         0.8280       593
   macro avg     0.8406    0.8443    0.8420       593
weighted avg     0.8266    0.8280    0.8269       593

```