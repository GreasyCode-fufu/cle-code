# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_dce4m_seed43/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8885 |
| Accuracy | 0.8786 |
| Macro-AUROC (OvR) | 0.9683 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8776 |
| Balanced Acc | 0.8931 |
| Macro-Sensitivity | 0.8931 |
| Macro-Specificity | 0.9331 |
| Macro-Precision | 0.8852 |
| AUC-PR (Macro) | 0.9537 |
| Cohen's Kappa | 0.8082 |
| MCC | 0.8086 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3721.0 (0.27 ms/张) |
| End-to-End FPS | 18.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8071 | 0.9318 | 0.8548 | 0.8303 | 0.9066 | 0.9530 | 0.9222 | 197 |
| normal | 0.8804 | 0.8864 | 0.8710 | 0.8757 | 0.8949 | 0.9534 | 0.9444 | 276 |
| sever | 0.9917 | 0.9810 | 0.9297 | 0.9597 | 0.9978 | 0.9987 | 0.9946 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=159 FP=27 TN=369 FN=38
- **normal**: TP=243 FP=36 TN=281 FN=33
- **sever**: TP=119 FP=9 TN=464 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 159 | 36 | 2 |
| normal | 26 | 243 | 7 |
| sever | 1 | 0 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8548    0.8071    0.8303       197
      normal     0.8710    0.8804    0.8757       276
       sever     0.9297    0.9917    0.9597       120

    accuracy                         0.8786       593
   macro avg     0.8852    0.8931    0.8885       593
weighted avg     0.8775    0.8786    0.8776       593

```