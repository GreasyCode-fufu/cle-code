# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce4s_seed42/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8537 |
| Accuracy | 0.8381 |
| Macro-AUROC (OvR) | 0.9463 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8384 |
| Balanced Acc | 0.8522 |
| Macro-Sensitivity | 0.8522 |
| Macro-Specificity | 0.9100 |
| Macro-Precision | 0.8552 |
| AUC-PR (Macro) | 0.9196 |
| Cohen's Kappa | 0.7436 |
| MCC | 0.7436 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4185.3 (0.24 ms/张) |
| End-to-End FPS | 20.1 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7817 | 0.8889 | 0.7778 | 0.7797 | 0.8911 | 0.9226 | 0.8652 | 197 |
| normal | 0.8333 | 0.8517 | 0.8303 | 0.8318 | 0.8544 | 0.9195 | 0.9048 | 276 |
| sever | 0.9417 | 0.9894 | 0.9576 | 0.9496 | 0.9853 | 0.9969 | 0.9889 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=154 FP=44 TN=352 FN=43
- **normal**: TP=230 FP=47 TN=270 FN=46
- **sever**: TP=113 FP=5 TN=468 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 154 | 43 | 0 |
| normal | 41 | 230 | 5 |
| sever | 3 | 4 | 113 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7778    0.7817    0.7797       197
      normal     0.8303    0.8333    0.8318       276
       sever     0.9576    0.9417    0.9496       120

    accuracy                         0.8381       593
   macro avg     0.8552    0.8522    0.8537       593
weighted avg     0.8386    0.8381    0.8384       593

```