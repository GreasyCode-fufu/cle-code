# convnext_wds_wtb_l1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wtb_l1_seed42/convnext_wds_wtb_l1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9795 |
| Accuracy | 0.9836 |
| Macro-AUROC (OvR) | 0.9982 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9837 |
| Balanced Acc | 0.9830 |
| Macro-Sensitivity | 0.9830 |
| Macro-Specificity | 0.9923 |
| Macro-Precision | 0.9762 |
| AUC-PR (Macro) | 0.9936 |
| Cohen's Kappa | 0.9740 |
| MCC | 0.9741 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.53G |
| FLOPs | 9.06G |
| Params | 28.53M |
| Pure-Inference FPS | 544.6 (1.84 ms/张) |
| End-to-End FPS | 192.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9914 | 0.9896 | 0.9819 | 0.9787 | 0.9981 | 0.9979 | 195 |
| Malignant_lesions | 0.9747 | 0.9856 | 0.9390 | 0.9565 | 0.9942 | 0.9964 | 0.9830 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=2 TN=230 FN=5
- **Malignant_lesions**: TP=77 FP=5 TN=343 FN=2
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 5 | 0 |
| Malignant_lesions | 2 | 77 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9896    0.9744    0.9819       195
     Malignant_lesions     0.9390    0.9747    0.9565        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9836       427
             macro avg     0.9762    0.9830    0.9795       427
          weighted avg     0.9840    0.9836    0.9837       427

```