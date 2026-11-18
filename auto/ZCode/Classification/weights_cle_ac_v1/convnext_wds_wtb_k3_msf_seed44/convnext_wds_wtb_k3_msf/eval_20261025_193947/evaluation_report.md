# convnext_wds_wtb_k3_msf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wtb_k3_msf_seed44/convnext_wds_wtb_k3_msf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9668 |
| Accuracy | 0.9670 |
| Macro-AUROC (OvR) | 0.9953 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9669 |
| Balanced Acc | 0.9671 |
| Macro-Sensitivity | 0.9671 |
| Macro-Specificity | 0.9835 |
| Macro-Precision | 0.9666 |
| AUC-PR (Macro) | 0.9915 |
| Cohen's Kappa | 0.9504 |
| MCC | 0.9505 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.53G |
| FLOPs | 9.05G |
| Params | 28.72M |
| Pure-Inference FPS | 415.5 (2.41 ms/张) |
| End-to-End FPS | 177.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9810 | 0.9659 | 0.9588 | 0.9730 | 0.9942 | 0.9888 | 208 |
| Malignant_lesions | 0.9545 | 0.9800 | 0.9545 | 0.9545 | 0.9800 | 0.9919 | 0.9860 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9896 | 0.9795 | 0.9871 | 0.9974 | 0.9998 | 0.9996 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=7 TN=361 FN=10
- **Malignant_lesions**: TP=168 FP=8 TN=392 FN=8
- **body_of_stomach_normal**: TP=191 FP=4 TN=380 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 8 | 2 |
| Malignant_lesions | 6 | 168 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9659    0.9519    0.9588       208
     Malignant_lesions     0.9545    0.9545    0.9545       176
body_of_stomach_normal     0.9795    0.9948    0.9871       192

              accuracy                         0.9670       576
             macro avg     0.9666    0.9671    0.9668       576
          weighted avg     0.9669    0.9670    0.9669       576

```