# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce16_seed44/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8395 |
| Accuracy | 0.8246 |
| Macro-AUROC (OvR) | 0.9302 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8235 |
| Balanced Acc | 0.8409 |
| Macro-Sensitivity | 0.8409 |
| Macro-Specificity | 0.9022 |
| Macro-Precision | 0.8390 |
| AUC-PR (Macro) | 0.8941 |
| Cohen's Kappa | 0.7220 |
| MCC | 0.7225 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 3915.3 (0.26 ms/张) |
| End-to-End FPS | 18.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7310 | 0.8990 | 0.7826 | 0.7559 | 0.8704 | 0.8989 | 0.8320 | 197 |
| normal | 0.8333 | 0.8265 | 0.8070 | 0.8200 | 0.8506 | 0.8965 | 0.8679 | 276 |
| sever | 0.9583 | 0.9810 | 0.9274 | 0.9426 | 0.9893 | 0.9951 | 0.9825 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=144 FP=40 TN=356 FN=53
- **normal**: TP=230 FP=55 TN=262 FN=46
- **sever**: TP=115 FP=9 TN=464 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 144 | 51 | 2 |
| normal | 39 | 230 | 7 |
| sever | 1 | 4 | 115 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7826    0.7310    0.7559       197
      normal     0.8070    0.8333    0.8200       276
       sever     0.9274    0.9583    0.9426       120

    accuracy                         0.8246       593
   macro avg     0.8390    0.8409    0.8395       593
weighted avg     0.8233    0.8246    0.8235       593

```