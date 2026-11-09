# fast_t0_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_dce4m_seed43/fast_t0_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9559 |
| Accuracy | 0.9649 |
| Macro-AUROC (OvR) | 0.9970 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9646 |
| Balanced Acc | 0.9518 |
| Macro-Sensitivity | 0.9518 |
| Macro-Specificity | 0.9811 |
| Macro-Precision | 0.9605 |
| AUC-PR (Macro) | 0.9911 |
| Cohen's Kappa | 0.9439 |
| MCC | 0.9440 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 4145.3 (0.24 ms/张) |
| End-to-End FPS | 7.9 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9612 | 0.9545 | 0.9618 | 0.9738 | 0.9968 | 0.9962 | 195 |
| Malignant_lesions | 0.8861 | 0.9856 | 0.9333 | 0.9091 | 0.9744 | 0.9943 | 0.9771 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=9 TN=223 FN=6
- **Malignant_lesions**: TP=70 FP=5 TN=343 FN=9
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 5 | 1 |
| Malignant_lesions | 9 | 70 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9545    0.9692    0.9618       195
     Malignant_lesions     0.9333    0.8861    0.9091        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9649       427
             macro avg     0.9605    0.9518    0.9559       427
          weighted avg     0.9646    0.9649    0.9646       427

```