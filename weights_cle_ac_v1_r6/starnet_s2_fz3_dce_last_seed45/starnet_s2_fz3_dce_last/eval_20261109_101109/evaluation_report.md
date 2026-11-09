# starnet_s2_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce_last_seed45/starnet_s2_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9208 |
| Accuracy | 0.9219 |
| Macro-AUROC (OvR) | 0.9888 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9212 |
| Balanced Acc | 0.9190 |
| Macro-Sensitivity | 0.9190 |
| Macro-Specificity | 0.9602 |
| Macro-Precision | 0.9258 |
| AUC-PR (Macro) | 0.9793 |
| Cohen's Kappa | 0.8822 |
| MCC | 0.8837 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 5489.8 (0.18 ms/张) |
| End-to-End FPS | 8.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9239 | 0.8744 | 0.9049 | 0.9632 | 0.9828 | 0.9735 | 208 |
| Malignant_lesions | 0.8352 | 0.9800 | 0.9484 | 0.8882 | 0.9311 | 0.9850 | 0.9671 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9766 | 0.9545 | 0.9692 | 0.9921 | 0.9987 | 0.9974 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=28 TN=340 FN=13
- **Malignant_lesions**: TP=147 FP=8 TN=392 FN=29
- **body_of_stomach_normal**: TP=189 FP=9 TN=375 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 8 | 5 |
| Malignant_lesions | 25 | 147 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8744    0.9375    0.9049       208
     Malignant_lesions     0.9484    0.8352    0.8882       176
body_of_stomach_normal     0.9545    0.9844    0.9692       192

              accuracy                         0.9219       576
             macro avg     0.9258    0.9190    0.9208       576
          weighted avg     0.9237    0.9219    0.9212       576

```