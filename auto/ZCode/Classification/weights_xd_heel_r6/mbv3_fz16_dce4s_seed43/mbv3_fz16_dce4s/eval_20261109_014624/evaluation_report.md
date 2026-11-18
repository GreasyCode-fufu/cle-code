# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce4s_seed43/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8642 |
| Accuracy | 0.8499 |
| Macro-AUROC (OvR) | 0.9469 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8500 |
| Balanced Acc | 0.8602 |
| Macro-Sensitivity | 0.8602 |
| Macro-Specificity | 0.9156 |
| Macro-Precision | 0.8687 |
| AUC-PR (Macro) | 0.9188 |
| Cohen's Kappa | 0.7615 |
| MCC | 0.7617 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3632.8 (0.28 ms/张) |
| End-to-End FPS | 18.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7766 | 0.9015 | 0.7969 | 0.7866 | 0.8903 | 0.9203 | 0.8716 | 197 |
| normal | 0.8623 | 0.8517 | 0.8351 | 0.8485 | 0.8766 | 0.9224 | 0.8933 | 276 |
| sever | 0.9417 | 0.9937 | 0.9741 | 0.9576 | 0.9853 | 0.9978 | 0.9915 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=153 FP=39 TN=357 FN=44
- **normal**: TP=238 FP=47 TN=270 FN=38
- **sever**: TP=113 FP=3 TN=470 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 153 | 44 | 0 |
| normal | 35 | 238 | 3 |
| sever | 4 | 3 | 113 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7969    0.7766    0.7866       197
      normal     0.8351    0.8623    0.8485       276
       sever     0.9741    0.9417    0.9576       120

    accuracy                         0.8499       593
   macro avg     0.8687    0.8602    0.8642       593
weighted avg     0.8505    0.8499    0.8500       593

```