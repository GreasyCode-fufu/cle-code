# starnet_s2_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce4m_seed43/starnet_s2_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9312 |
| Accuracy | 0.9323 |
| Macro-AUROC (OvR) | 0.9879 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9317 |
| Balanced Acc | 0.9310 |
| Macro-Sensitivity | 0.9310 |
| Macro-Specificity | 0.9659 |
| Macro-Precision | 0.9322 |
| AUC-PR (Macro) | 0.9760 |
| Cohen's Kappa | 0.8981 |
| MCC | 0.8985 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3456.5 (0.29 ms/张) |
| End-to-End FPS | 7.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9538 | 0.9187 | 0.9209 | 0.9564 | 0.9812 | 0.9730 | 208 |
| Malignant_lesions | 0.8750 | 0.9700 | 0.9277 | 0.9006 | 0.9463 | 0.9833 | 0.9570 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9740 | 0.9502 | 0.9720 | 0.9973 | 0.9990 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=17 TN=351 FN=16
- **Malignant_lesions**: TP=154 FP=12 TN=388 FN=22
- **body_of_stomach_normal**: TP=191 FP=10 TN=374 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 12 | 4 |
| Malignant_lesions | 16 | 154 | 6 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9187    0.9231    0.9209       208
     Malignant_lesions     0.9277    0.8750    0.9006       176
body_of_stomach_normal     0.9502    0.9948    0.9720       192

              accuracy                         0.9323       576
             macro avg     0.9322    0.9310    0.9312       576
          weighted avg     0.9320    0.9323    0.9317       576

```