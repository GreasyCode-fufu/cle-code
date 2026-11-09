# starnet_s2_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce4m_seed42/starnet_s2_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9448 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9937 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9446 |
| Balanced Acc | 0.9439 |
| Macro-Sensitivity | 0.9439 |
| Macro-Specificity | 0.9718 |
| Macro-Precision | 0.9461 |
| AUC-PR (Macro) | 0.9877 |
| Cohen's Kappa | 0.9164 |
| MCC | 0.9166 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3684.2 (0.27 ms/张) |
| End-to-End FPS | 7.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9484 | 0.9120 | 0.9292 | 0.9694 | 0.9900 | 0.9848 | 208 |
| Malignant_lesions | 0.9261 | 0.9775 | 0.9477 | 0.9368 | 0.9678 | 0.9926 | 0.9812 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9896 | 0.9787 | 0.9684 | 0.9794 | 0.9984 | 0.9970 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=19 TN=349 FN=11
- **Malignant_lesions**: TP=163 FP=9 TN=391 FN=13
- **body_of_stomach_normal**: TP=184 FP=4 TN=380 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 11 | 163 | 2 |
| body_of_stomach_normal | 8 | 0 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9120    0.9471    0.9292       208
     Malignant_lesions     0.9477    0.9261    0.9368       176
body_of_stomach_normal     0.9787    0.9583    0.9684       192

              accuracy                         0.9444       576
             macro avg     0.9461    0.9439    0.9448       576
          weighted avg     0.9452    0.9444    0.9446       576

```