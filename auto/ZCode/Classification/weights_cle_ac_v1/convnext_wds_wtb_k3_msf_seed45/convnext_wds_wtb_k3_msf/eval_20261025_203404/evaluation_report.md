# convnext_wds_wtb_k3_msf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wtb_k3_msf_seed45/convnext_wds_wtb_k3_msf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9666 |
| Accuracy | 0.9670 |
| Macro-AUROC (OvR) | 0.9975 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9670 |
| Balanced Acc | 0.9657 |
| Macro-Sensitivity | 0.9657 |
| Macro-Specificity | 0.9832 |
| Macro-Precision | 0.9681 |
| AUC-PR (Macro) | 0.9953 |
| Cohen's Kappa | 0.9504 |
| MCC | 0.9506 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.53G |
| FLOPs | 9.05G |
| Params | 28.72M |
| Pure-Inference FPS | 1155.7 (0.87 ms/张) |
| End-to-End FPS | 247.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9674 | 0.9444 | 0.9623 | 0.9889 | 0.9971 | 0.9948 | 208 |
| Malignant_lesions | 0.9318 | 0.9875 | 0.9704 | 0.9507 | 0.9705 | 0.9959 | 0.9920 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9948 | 0.9895 | 0.9869 | 0.9922 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=12 TN=356 FN=4
- **Malignant_lesions**: TP=164 FP=5 TN=395 FN=12
- **body_of_stomach_normal**: TP=189 FP=2 TN=382 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 4 | 0 |
| Malignant_lesions | 10 | 164 | 2 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9444    0.9808    0.9623       208
     Malignant_lesions     0.9704    0.9318    0.9507       176
body_of_stomach_normal     0.9895    0.9844    0.9869       192

              accuracy                         0.9670       576
             macro avg     0.9681    0.9657    0.9666       576
          weighted avg     0.9674    0.9670    0.9670       576

```