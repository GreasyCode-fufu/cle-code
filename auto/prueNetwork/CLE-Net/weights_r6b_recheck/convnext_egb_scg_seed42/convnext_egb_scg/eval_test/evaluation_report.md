# convnext_egb_scg 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/weights_r6b_recheck/convnext_egb_scg_seed42/convnext_egb_scg/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9534 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9954 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9532 |
| Balanced Acc | 0.9531 |
| Macro-Sensitivity | 0.9531 |
| Macro-Specificity | 0.9763 |
| Macro-Precision | 0.9546 |
| AUC-PR (Macro) | 0.9921 |
| Cohen's Kappa | 0.9295 |
| MCC | 0.9300 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 469.5 (2.13 ms/张) |
| End-to-End FPS | 12.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9592 | 0.9306 | 0.9481 | 0.9806 | 0.9934 | 0.9895 | 208 |
| Malignant_lesions | 0.9659 | 0.9750 | 0.9444 | 0.9551 | 0.9848 | 0.9936 | 0.9884 | 176 |
| body_of_stomach_normal | 0.9271 | 0.9948 | 0.9889 | 0.9570 | 0.9646 | 0.9992 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=15 TN=353 FN=7
- **Malignant_lesions**: TP=170 FP=10 TN=390 FN=6
- **body_of_stomach_normal**: TP=178 FP=2 TN=382 FN=14

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 5 | 170 | 1 |
| body_of_stomach_normal | 10 | 4 | 178 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9306    0.9663    0.9481       208
     Malignant_lesions     0.9444    0.9659    0.9551       176
body_of_stomach_normal     0.9889    0.9271    0.9570       192

              accuracy                         0.9531       576
             macro avg     0.9546    0.9531    0.9534       576
          weighted avg     0.9542    0.9531    0.9532       576

```