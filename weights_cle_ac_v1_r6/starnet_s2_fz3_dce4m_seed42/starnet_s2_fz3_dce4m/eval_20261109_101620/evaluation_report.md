# starnet_s2_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce4m_seed42/starnet_s2_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9167 |
| Accuracy | 0.9184 |
| Macro-AUROC (OvR) | 0.9887 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9175 |
| Balanced Acc | 0.9155 |
| Macro-Sensitivity | 0.9155 |
| Macro-Specificity | 0.9586 |
| Macro-Precision | 0.9207 |
| AUC-PR (Macro) | 0.9777 |
| Cohen's Kappa | 0.8770 |
| MCC | 0.8783 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 4343.8 (0.23 ms/张) |
| End-to-End FPS | 8.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9321 | 0.8858 | 0.9087 | 0.9608 | 0.9821 | 0.9752 | 208 |
| Malignant_lesions | 0.8295 | 0.9750 | 0.9359 | 0.8795 | 0.9286 | 0.9855 | 0.9607 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9688 | 0.9403 | 0.9618 | 0.9920 | 0.9986 | 0.9971 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=25 TN=343 FN=14
- **Malignant_lesions**: TP=146 FP=10 TN=390 FN=30
- **body_of_stomach_normal**: TP=189 FP=12 TN=372 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 10 | 4 |
| Malignant_lesions | 22 | 146 | 8 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8858    0.9327    0.9087       208
     Malignant_lesions     0.9359    0.8295    0.8795       176
body_of_stomach_normal     0.9403    0.9844    0.9618       192

              accuracy                         0.9184       576
             macro avg     0.9207    0.9155    0.9167       576
          weighted avg     0.9193    0.9184    0.9175       576

```