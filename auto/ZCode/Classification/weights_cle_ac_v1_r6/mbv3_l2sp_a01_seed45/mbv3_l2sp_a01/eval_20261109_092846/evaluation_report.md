# mbv3_l2sp_a01 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_l2sp_a01_seed45/mbv3_l2sp_a01/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9328 |
| Accuracy | 0.9323 |
| Macro-AUROC (OvR) | 0.9917 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9325 |
| Balanced Acc | 0.9299 |
| Macro-Sensitivity | 0.9299 |
| Macro-Specificity | 0.9652 |
| Macro-Precision | 0.9391 |
| AUC-PR (Macro) | 0.9845 |
| Cohen's Kappa | 0.8979 |
| MCC | 0.8996 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2641.7 (0.38 ms/张) |
| End-to-End FPS | 145.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9185 | 0.8690 | 0.9108 | 0.9741 | 0.9868 | 0.9780 | 208 |
| Malignant_lesions | 0.8693 | 0.9900 | 0.9745 | 0.9189 | 0.9451 | 0.9901 | 0.9789 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9870 | 0.9737 | 0.9686 | 0.9819 | 0.9982 | 0.9966 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=30 TN=338 FN=9
- **Malignant_lesions**: TP=153 FP=4 TN=396 FN=23
- **body_of_stomach_normal**: TP=185 FP=5 TN=379 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 4 | 5 |
| Malignant_lesions | 23 | 153 | 0 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8690    0.9567    0.9108       208
     Malignant_lesions     0.9745    0.8693    0.9189       176
body_of_stomach_normal     0.9737    0.9635    0.9686       192

              accuracy                         0.9323       576
             macro avg     0.9391    0.9299    0.9328       576
          weighted avg     0.9361    0.9323    0.9325       576

```