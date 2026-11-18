# convnext_wds_wdd_msf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wdd_msf_seed44/convnext_wds_wdd_msf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9759 |
| Accuracy | 0.9813 |
| Macro-AUROC (OvR) | 0.9985 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9811 |
| Balanced Acc | 0.9713 |
| Macro-Sensitivity | 0.9713 |
| Macro-Specificity | 0.9895 |
| Macro-Precision | 0.9811 |
| AUC-PR (Macro) | 0.9957 |
| Cohen's Kappa | 0.9701 |
| MCC | 0.9702 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.68G |
| FLOPs | 9.36G |
| Params | 29.80M |
| Pure-Inference FPS | 446.5 (2.24 ms/张) |
| End-to-End FPS | 183.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9897 | 0.9741 | 0.9698 | 0.9797 | 0.9912 | 0.9982 | 0.9979 | 195 |
| Malignant_lesions | 0.9241 | 0.9943 | 0.9733 | 0.9481 | 0.9830 | 0.9972 | 0.9892 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=6 TN=226 FN=2
- **Malignant_lesions**: TP=73 FP=2 TN=346 FN=6
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 2 | 0 |
| Malignant_lesions | 6 | 73 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9698    0.9897    0.9797       195
     Malignant_lesions     0.9733    0.9241    0.9481        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9813       427
             macro avg     0.9811    0.9713    0.9759       427
          weighted avg     0.9813    0.9813    0.9811       427

```