# starnet_s2_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce4m_seed42/starnet_s2_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9510 |
| Accuracy | 0.9602 |
| Macro-AUROC (OvR) | 0.9956 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9604 |
| Balanced Acc | 0.9559 |
| Macro-Sensitivity | 0.9559 |
| Macro-Specificity | 0.9808 |
| Macro-Precision | 0.9467 |
| AUC-PR (Macro) | 0.9857 |
| Cohen's Kappa | 0.9370 |
| MCC | 0.9372 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3716.0 (0.27 ms/张) |
| End-to-End FPS | 8.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9436 | 0.9784 | 0.9735 | 0.9583 | 0.9538 | 0.9950 | 0.9946 | 195 |
| Malignant_lesions | 0.9241 | 0.9713 | 0.8795 | 0.9012 | 0.9826 | 0.9919 | 0.9624 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=184 FP=5 TN=227 FN=11
- **Malignant_lesions**: TP=73 FP=10 TN=338 FN=6
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 184 | 10 | 1 |
| Malignant_lesions | 5 | 73 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9735    0.9436    0.9583       195
     Malignant_lesions     0.8795    0.9241    0.9012        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9602       427
             macro avg     0.9467    0.9559    0.9510       427
          weighted avg     0.9610    0.9602    0.9604       427

```