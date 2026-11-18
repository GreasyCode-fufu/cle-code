# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce4s_seed42/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8311 |
| Accuracy | 0.8162 |
| Macro-AUROC (OvR) | 0.9343 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8150 |
| Balanced Acc | 0.8339 |
| Macro-Sensitivity | 0.8339 |
| Macro-Specificity | 0.8983 |
| Macro-Precision | 0.8291 |
| AUC-PR (Macro) | 0.8987 |
| Cohen's Kappa | 0.7092 |
| MCC | 0.7094 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3706.4 (0.27 ms/张) |
| End-to-End FPS | 18.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7208 | 0.8864 | 0.7594 | 0.7396 | 0.8645 | 0.9031 | 0.8238 | 197 |
| normal | 0.8225 | 0.8297 | 0.8078 | 0.8151 | 0.8429 | 0.9035 | 0.8870 | 276 |
| sever | 0.9583 | 0.9789 | 0.9200 | 0.9388 | 0.9893 | 0.9963 | 0.9854 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=142 FP=45 TN=351 FN=55
- **normal**: TP=227 FP=54 TN=263 FN=49
- **sever**: TP=115 FP=10 TN=463 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 142 | 51 | 4 |
| normal | 43 | 227 | 6 |
| sever | 2 | 3 | 115 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7594    0.7208    0.7396       197
      normal     0.8078    0.8225    0.8151       276
       sever     0.9200    0.9583    0.9388       120

    accuracy                         0.8162       593
   macro avg     0.8291    0.8339    0.8311       593
weighted avg     0.8144    0.8162    0.8150       593

```