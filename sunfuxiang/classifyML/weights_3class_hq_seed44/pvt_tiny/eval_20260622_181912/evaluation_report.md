# pvt_tiny 评估报告

- 权重: `weights_3class_hq_seed44/pvt_tiny/pvt_tiny_3class_hq_seed44_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9032 |
| Accuracy | 0.9016 |
| Macro-AUROC (OvR) | 0.9751 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9018 |
| Balanced Acc | 0.9038 |
| Macro-Sensitivity | 0.9038 |
| Macro-Specificity | 0.9504 |
| Macro-Precision | 0.9032 |
| AUC-PR (Macro) | 0.9542 |
| Cohen's Kappa | 0.8518 |
| MCC | 0.8521 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2907.8 (0.34 ms/张) |
| End-to-End FPS | 228.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8750 | 0.9398 | 0.8996 | 0.8871 | 0.9242 | 0.9682 | 0.9546 | 256 |
| Malignant_lesions | 0.9048 | 0.9328 | 0.8597 | 0.8817 | 0.9556 | 0.9645 | 0.9245 | 210 |
| body_of_stomach_normal | 0.9317 | 0.9785 | 0.9502 | 0.9409 | 0.9702 | 0.9925 | 0.9834 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=224 FP=25 TN=390 FN=32
- **Malignant_lesions**: TP=190 FP=31 TN=430 FN=20
- **body_of_stomach_normal**: TP=191 FP=10 TN=456 FN=14

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 224 | 24 | 8 |
| Malignant_lesions | 18 | 190 | 2 |
| body_of_stomach_normal | 7 | 7 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8996    0.8750    0.8871       256
     Malignant_lesions     0.8597    0.9048    0.8817       210
body_of_stomach_normal     0.9502    0.9317    0.9409       205

              accuracy                         0.9016       671
             macro avg     0.9032    0.9038    0.9032       671
          weighted avg     0.9026    0.9016    0.9018       671

```