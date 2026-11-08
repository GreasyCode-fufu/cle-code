# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_seed46/fast_t0/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9591 |
| Accuracy | 0.9672 |
| Macro-AUROC (OvR) | 0.9945 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9670 |
| Balanced Acc | 0.9560 |
| Macro-Sensitivity | 0.9560 |
| Macro-Specificity | 0.9825 |
| Macro-Precision | 0.9624 |
| AUC-PR (Macro) | 0.9839 |
| Cohen's Kappa | 0.9477 |
| MCC | 0.9478 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 5877.5 (0.17 ms/张) |
| End-to-End FPS | 87.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9655 | 0.9594 | 0.9643 | 0.9739 | 0.9935 | 0.9926 | 195 |
| Malignant_lesions | 0.8987 | 0.9856 | 0.9342 | 0.9161 | 0.9772 | 0.9901 | 0.9590 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=8 TN=224 FN=6
- **Malignant_lesions**: TP=71 FP=5 TN=343 FN=8
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 5 | 1 |
| Malignant_lesions | 8 | 71 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9594    0.9692    0.9643       195
     Malignant_lesions     0.9342    0.8987    0.9161        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9672       427
             macro avg     0.9624    0.9560    0.9591       427
          weighted avg     0.9670    0.9672    0.9670       427

```