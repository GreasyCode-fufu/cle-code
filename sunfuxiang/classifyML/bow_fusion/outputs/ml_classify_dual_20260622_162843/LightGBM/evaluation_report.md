# LightGBM 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca-2/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8893 |
| Accuracy | 0.8882 |
| Macro-AUROC (OvR) | 0.9733 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8873 |
| Balanced Acc | 0.8925 |
| Macro-Sensitivity | 0.8925 |
| Macro-Specificity | 0.9439 |
| Macro-Precision | 0.8879 |
| AUC-PR (Macro) | 0.9585 |
| Cohen's Kappa | 0.8319 |
| MCC | 0.8327 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 372.86M |
| FLOPs | 745.72M |
| Params | 3.69M |
| Pure-Inference FPS | 345534.4 (0.00 ms/张) |
| End-to-End FPS | 793.6 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8398 | 0.9373 | 0.8921 | 0.8652 | 0.9047 | 0.9601 | 0.9471 | 256 |
| Malignant_lesions | 0.8619 | 0.9458 | 0.8786 | 0.8702 | 0.9376 | 0.9655 | 0.9402 | 210 |
| body_of_stomach_normal | 0.9756 | 0.9485 | 0.8929 | 0.9324 | 0.9888 | 0.9942 | 0.9884 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=215 FP=26 TN=389 FN=41
- **Malignant_lesions**: TP=181 FP=25 TN=436 FN=29
- **body_of_stomach_normal**: TP=200 FP=24 TN=442 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 215 | 23 | 18 |
| Malignant_lesions | 23 | 181 | 6 |
| body_of_stomach_normal | 3 | 2 | 200 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8921    0.8398    0.8652       256
     Malignant_lesions     0.8786    0.8619    0.8702       210
body_of_stomach_normal     0.8929    0.9756    0.9324       205

              accuracy                         0.8882       671
             macro avg     0.8879    0.8925    0.8893       671
          weighted avg     0.8881    0.8882    0.8873       671

```