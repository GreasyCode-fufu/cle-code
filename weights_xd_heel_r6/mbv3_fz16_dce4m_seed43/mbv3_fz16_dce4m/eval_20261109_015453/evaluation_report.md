# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce4m_seed43/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8691 |
| Accuracy | 0.8550 |
| Macro-AUROC (OvR) | 0.9467 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8548 |
| Balanced Acc | 0.8654 |
| Macro-Sensitivity | 0.8654 |
| Macro-Specificity | 0.9182 |
| Macro-Precision | 0.8734 |
| AUC-PR (Macro) | 0.9167 |
| Cohen's Kappa | 0.7694 |
| MCC | 0.7698 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3684.3 (0.27 ms/张) |
| End-to-End FPS | 19.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7766 | 0.9091 | 0.8095 | 0.7927 | 0.8911 | 0.9197 | 0.8697 | 197 |
| normal | 0.8696 | 0.8517 | 0.8362 | 0.8526 | 0.8824 | 0.9229 | 0.8924 | 276 |
| sever | 0.9500 | 0.9937 | 0.9744 | 0.9620 | 0.9874 | 0.9974 | 0.9881 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=153 FP=36 TN=360 FN=44
- **normal**: TP=240 FP=47 TN=270 FN=36
- **sever**: TP=114 FP=3 TN=470 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 153 | 44 | 0 |
| normal | 33 | 240 | 3 |
| sever | 3 | 3 | 114 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8095    0.7766    0.7927       197
      normal     0.8362    0.8696    0.8526       276
       sever     0.9744    0.9500    0.9620       120

    accuracy                         0.8550       593
   macro avg     0.8734    0.8654    0.8691       593
weighted avg     0.8553    0.8550    0.8548       593

```