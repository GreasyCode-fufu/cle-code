# starnet_s2_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce4m_seed46/starnet_s2_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9439 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9942 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9442 |
| Balanced Acc | 0.9423 |
| Macro-Sensitivity | 0.9423 |
| Macro-Specificity | 0.9716 |
| Macro-Precision | 0.9472 |
| AUC-PR (Macro) | 0.9895 |
| Cohen's Kappa | 0.9163 |
| MCC | 0.9171 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 5241.9 (0.19 ms/张) |
| End-to-End FPS | 7.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9429 | 0.9045 | 0.9299 | 0.9747 | 0.9915 | 0.9853 | 208 |
| Malignant_lesions | 0.8807 | 0.9850 | 0.9627 | 0.9199 | 0.9494 | 0.9918 | 0.9848 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9991 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=21 TN=347 FN=9
- **Malignant_lesions**: TP=155 FP=6 TN=394 FN=21
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 6 | 3 |
| Malignant_lesions | 19 | 155 | 2 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9045    0.9567    0.9299       208
     Malignant_lesions     0.9627    0.8807    0.9199       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9444       576
             macro avg     0.9472    0.9423    0.9439       576
          weighted avg     0.9456    0.9444    0.9442       576

```