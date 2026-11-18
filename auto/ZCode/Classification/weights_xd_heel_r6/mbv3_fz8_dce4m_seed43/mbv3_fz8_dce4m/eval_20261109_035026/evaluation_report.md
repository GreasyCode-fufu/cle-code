# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz8_dce4m_seed43/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8652 |
| Accuracy | 0.8550 |
| Macro-AUROC (OvR) | 0.9536 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8541 |
| Balanced Acc | 0.8619 |
| Macro-Sensitivity | 0.8619 |
| Macro-Specificity | 0.9182 |
| Macro-Precision | 0.8702 |
| AUC-PR (Macro) | 0.9318 |
| Cohen's Kappa | 0.7690 |
| MCC | 0.7701 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4003.4 (0.25 ms/张) |
| End-to-End FPS | 13.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7563 | 0.9217 | 0.8278 | 0.7905 | 0.8838 | 0.9308 | 0.8874 | 197 |
| normal | 0.8877 | 0.8454 | 0.8333 | 0.8596 | 0.8963 | 0.9332 | 0.9178 | 276 |
| sever | 0.9417 | 0.9873 | 0.9496 | 0.9456 | 0.9852 | 0.9969 | 0.9901 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=149 FP=31 TN=365 FN=48
- **normal**: TP=245 FP=49 TN=268 FN=31
- **sever**: TP=113 FP=6 TN=467 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 149 | 46 | 2 |
| normal | 27 | 245 | 4 |
| sever | 4 | 3 | 113 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8278    0.7563    0.7905       197
      normal     0.8333    0.8877    0.8596       276
       sever     0.9496    0.9417    0.9456       120

    accuracy                         0.8550       593
   macro avg     0.8702    0.8619    0.8652       593
weighted avg     0.8550    0.8550    0.8541       593

```