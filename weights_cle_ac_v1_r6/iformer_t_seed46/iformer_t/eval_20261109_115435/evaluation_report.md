# iformer_t 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/iformer_t_seed46/iformer_t/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9389 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9925 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9392 |
| Balanced Acc | 0.9366 |
| Macro-Sensitivity | 0.9366 |
| Macro-Specificity | 0.9688 |
| Macro-Precision | 0.9440 |
| AUC-PR (Macro) | 0.9868 |
| Cohen's Kappa | 0.9084 |
| MCC | 0.9098 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 521.61M |
| FLOPs | 1.04G |
| Params | 2.63M |
| Pure-Inference FPS | 6029.0 (0.17 ms/张) |
| End-to-End FPS | 222.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9321 | 0.8899 | 0.9287 | 0.9828 | 0.9906 | 0.9837 | 208 |
| Malignant_lesions | 0.8750 | 0.9875 | 0.9686 | 0.9194 | 0.9472 | 0.9887 | 0.9803 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9870 | 0.9737 | 0.9686 | 0.9819 | 0.9982 | 0.9964 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=25 TN=343 FN=6
- **Malignant_lesions**: TP=154 FP=5 TN=395 FN=22
- **body_of_stomach_normal**: TP=185 FP=5 TN=379 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 2 | 4 |
| Malignant_lesions | 21 | 154 | 1 |
| body_of_stomach_normal | 4 | 3 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8899    0.9712    0.9287       208
     Malignant_lesions     0.9686    0.8750    0.9194       176
body_of_stomach_normal     0.9737    0.9635    0.9686       192

              accuracy                         0.9392       576
             macro avg     0.9440    0.9366    0.9389       576
          weighted avg     0.9418    0.9392    0.9392       576

```