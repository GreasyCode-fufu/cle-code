# starnet_s2_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce_last_seed46/starnet_s2_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9627 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9960 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9693 |
| Balanced Acc | 0.9568 |
| Macro-Sensitivity | 0.9568 |
| Macro-Specificity | 0.9830 |
| Macro-Precision | 0.9694 |
| AUC-PR (Macro) | 0.9899 |
| Cohen's Kappa | 0.9513 |
| MCC | 0.9516 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3464.8 (0.29 ms/张) |
| End-to-End FPS | 7.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9612 | 0.9552 | 0.9697 | 0.9867 | 0.9960 | 0.9949 | 195 |
| Malignant_lesions | 0.8987 | 0.9914 | 0.9595 | 0.9281 | 0.9773 | 0.9921 | 0.9750 | 79 |
| body_of_stomach_normal | 0.9869 | 0.9964 | 0.9934 | 0.9902 | 0.9927 | 0.9998 | 0.9997 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=9 TN=223 FN=3
- **Malignant_lesions**: TP=71 FP=3 TN=345 FN=8
- **body_of_stomach_normal**: TP=151 FP=1 TN=273 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 3 | 0 |
| Malignant_lesions | 7 | 71 | 1 |
| body_of_stomach_normal | 2 | 0 | 151 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9552    0.9846    0.9697       195
     Malignant_lesions     0.9595    0.8987    0.9281        79
body_of_stomach_normal     0.9934    0.9869    0.9902       153

              accuracy                         0.9696       427
             macro avg     0.9694    0.9568    0.9627       427
          weighted avg     0.9697    0.9696    0.9693       427

```