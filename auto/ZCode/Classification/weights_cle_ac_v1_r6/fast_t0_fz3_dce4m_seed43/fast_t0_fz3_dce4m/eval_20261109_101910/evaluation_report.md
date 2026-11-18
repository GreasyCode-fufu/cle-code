# fast_t0_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_dce4m_seed43/fast_t0_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9420 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9946 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9424 |
| Balanced Acc | 0.9406 |
| Macro-Sensitivity | 0.9406 |
| Macro-Specificity | 0.9708 |
| Macro-Precision | 0.9449 |
| AUC-PR (Macro) | 0.9893 |
| Cohen's Kappa | 0.9137 |
| MCC | 0.9144 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 6681.5 (0.15 ms/张) |
| End-to-End FPS | 8.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9457 | 0.9087 | 0.9321 | 0.9748 | 0.9912 | 0.9865 | 208 |
| Malignant_lesions | 0.8807 | 0.9825 | 0.9568 | 0.9172 | 0.9493 | 0.9934 | 0.9827 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9844 | 0.9692 | 0.9767 | 0.9921 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=20 TN=348 FN=9
- **Malignant_lesions**: TP=155 FP=7 TN=393 FN=21
- **body_of_stomach_normal**: TP=189 FP=6 TN=378 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 17 | 155 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9087    0.9567    0.9321       208
     Malignant_lesions     0.9568    0.8807    0.9172       176
body_of_stomach_normal     0.9692    0.9844    0.9767       192

              accuracy                         0.9427       576
             macro avg     0.9449    0.9406    0.9420       576
          weighted avg     0.9436    0.9427    0.9424       576

```