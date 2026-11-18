# convnext_wds_wtb_k3_msf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wtb_k3_msf_seed43/convnext_wds_wtb_k3_msf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9510 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9955 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9513 |
| Balanced Acc | 0.9512 |
| Macro-Sensitivity | 0.9512 |
| Macro-Specificity | 0.9757 |
| Macro-Precision | 0.9509 |
| AUC-PR (Macro) | 0.9914 |
| Cohen's Kappa | 0.9269 |
| MCC | 0.9269 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.53G |
| FLOPs | 9.05G |
| Params | 28.72M |
| Pure-Inference FPS | 662.9 (1.51 ms/张) |
| End-to-End FPS | 212.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9701 | 0.9466 | 0.9420 | 0.9649 | 0.9937 | 0.9887 | 208 |
| Malignant_lesions | 0.9318 | 0.9700 | 0.9318 | 0.9318 | 0.9700 | 0.9938 | 0.9871 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9870 | 0.9742 | 0.9793 | 0.9921 | 0.9991 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=11 TN=357 FN=13
- **Malignant_lesions**: TP=164 FP=12 TN=388 FN=12
- **body_of_stomach_normal**: TP=189 FP=5 TN=379 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 12 | 1 |
| Malignant_lesions | 8 | 164 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9466    0.9375    0.9420       208
     Malignant_lesions     0.9318    0.9318    0.9318       176
body_of_stomach_normal     0.9742    0.9844    0.9793       192

              accuracy                         0.9514       576
             macro avg     0.9509    0.9512    0.9510       576
          weighted avg     0.9513    0.9514    0.9513       576

```