# starnet_s2_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce4m_seed46/starnet_s2_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9653 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9964 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9716 |
| Balanced Acc | 0.9585 |
| Macro-Sensitivity | 0.9585 |
| Macro-Specificity | 0.9842 |
| Macro-Precision | 0.9732 |
| AUC-PR (Macro) | 0.9912 |
| Cohen's Kappa | 0.9550 |
| MCC | 0.9554 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3091.2 (0.32 ms/张) |
| End-to-End FPS | 7.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9897 | 0.9655 | 0.9602 | 0.9747 | 0.9912 | 0.9964 | 0.9953 | 195 |
| Malignant_lesions | 0.8987 | 0.9943 | 0.9726 | 0.9342 | 0.9774 | 0.9932 | 0.9786 | 79 |
| body_of_stomach_normal | 0.9869 | 0.9927 | 0.9869 | 0.9869 | 0.9927 | 0.9998 | 0.9996 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=8 TN=224 FN=2
- **Malignant_lesions**: TP=71 FP=2 TN=346 FN=8
- **body_of_stomach_normal**: TP=151 FP=2 TN=272 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 2 | 0 |
| Malignant_lesions | 6 | 71 | 2 |
| body_of_stomach_normal | 2 | 0 | 151 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9602    0.9897    0.9747       195
     Malignant_lesions     0.9726    0.8987    0.9342        79
body_of_stomach_normal     0.9869    0.9869    0.9869       153

              accuracy                         0.9719       427
             macro avg     0.9732    0.9585    0.9653       427
          weighted avg     0.9721    0.9719    0.9716       427

```