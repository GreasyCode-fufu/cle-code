# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/lsnet_s_seed42/lsnet_s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9197 |
| Accuracy | 0.9201 |
| Macro-AUROC (OvR) | 0.9890 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9200 |
| Balanced Acc | 0.9182 |
| Macro-Sensitivity | 0.9182 |
| Macro-Specificity | 0.9595 |
| Macro-Precision | 0.9224 |
| AUC-PR (Macro) | 0.9788 |
| Cohen's Kappa | 0.8797 |
| MCC | 0.8803 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 3697.7 (0.27 ms/张) |
| End-to-End FPS | 233.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9321 | 0.8864 | 0.9112 | 0.9635 | 0.9838 | 0.9765 | 208 |
| Malignant_lesions | 0.8693 | 0.9725 | 0.9329 | 0.9000 | 0.9442 | 0.9858 | 0.9654 | 176 |
| body_of_stomach_normal | 0.9479 | 0.9740 | 0.9479 | 0.9479 | 0.9740 | 0.9972 | 0.9944 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=25 TN=343 FN=13
- **Malignant_lesions**: TP=153 FP=11 TN=389 FN=23
- **body_of_stomach_normal**: TP=182 FP=10 TN=374 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 9 | 4 |
| Malignant_lesions | 17 | 153 | 6 |
| body_of_stomach_normal | 8 | 2 | 182 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8864    0.9375    0.9112       208
     Malignant_lesions     0.9329    0.8693    0.9000       176
body_of_stomach_normal     0.9479    0.9479    0.9479       192

              accuracy                         0.9201       576
             macro avg     0.9224    0.9182    0.9197       576
          weighted avg     0.9211    0.9201    0.9200       576

```