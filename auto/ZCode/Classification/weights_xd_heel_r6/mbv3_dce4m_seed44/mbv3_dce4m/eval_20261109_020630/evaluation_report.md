# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_dce4m_seed44/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8859 |
| Accuracy | 0.8752 |
| Macro-AUROC (OvR) | 0.9659 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8742 |
| Balanced Acc | 0.8876 |
| Macro-Sensitivity | 0.8876 |
| Macro-Specificity | 0.9303 |
| Macro-Precision | 0.8854 |
| AUC-PR (Macro) | 0.9498 |
| Cohen's Kappa | 0.8023 |
| MCC | 0.8028 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3704.4 (0.27 ms/张) |
| End-to-End FPS | 18.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7919 | 0.9318 | 0.8525 | 0.8211 | 0.9000 | 0.9482 | 0.9149 | 197 |
| normal | 0.8877 | 0.8738 | 0.8596 | 0.8734 | 0.8994 | 0.9508 | 0.9396 | 276 |
| sever | 0.9833 | 0.9852 | 0.9440 | 0.9633 | 0.9957 | 0.9987 | 0.9950 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=156 FP=27 TN=369 FN=41
- **normal**: TP=245 FP=40 TN=277 FN=31
- **sever**: TP=118 FP=7 TN=466 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 156 | 39 | 2 |
| normal | 26 | 245 | 5 |
| sever | 1 | 1 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8525    0.7919    0.8211       197
      normal     0.8596    0.8877    0.8734       276
       sever     0.9440    0.9833    0.9633       120

    accuracy                         0.8752       593
   macro avg     0.8854    0.8876    0.8859       593
weighted avg     0.8743    0.8752    0.8742       593

```