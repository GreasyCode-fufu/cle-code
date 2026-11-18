# convnext_fgc_g8 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_g8_seed43/convnext_fgc_g8/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9634 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9634 |
| Balanced Acc | 0.9642 |
| Macro-Sensitivity | 0.9642 |
| Macro-Specificity | 0.9819 |
| Macro-Precision | 0.9629 |
| AUC-PR (Macro) | 0.9934 |
| Cohen's Kappa | 0.9452 |
| MCC | 0.9454 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 630.8 (1.59 ms/张) |
| End-to-End FPS | 202.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9837 | 0.9701 | 0.9535 | 0.9653 | 0.9953 | 0.9929 | 208 |
| Malignant_lesions | 0.9602 | 0.9750 | 0.9441 | 0.9521 | 0.9824 | 0.9952 | 0.9880 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9870 | 0.9745 | 0.9845 | 0.9974 | 0.9997 | 0.9993 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=6 TN=362 FN=13
- **Malignant_lesions**: TP=169 FP=10 TN=390 FN=7
- **body_of_stomach_normal**: TP=191 FP=5 TN=379 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 10 | 3 |
| Malignant_lesions | 5 | 169 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9701    0.9375    0.9535       208
     Malignant_lesions     0.9441    0.9602    0.9521       176
body_of_stomach_normal     0.9745    0.9948    0.9845       192

              accuracy                         0.9635       576
             macro avg     0.9629    0.9642    0.9634       576
          weighted avg     0.9636    0.9635    0.9634       576

```