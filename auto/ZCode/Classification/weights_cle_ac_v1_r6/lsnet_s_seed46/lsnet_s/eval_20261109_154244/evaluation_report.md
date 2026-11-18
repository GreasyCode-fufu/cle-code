# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/lsnet_s_seed46/lsnet_s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9124 |
| Accuracy | 0.9132 |
| Macro-AUROC (OvR) | 0.9848 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9128 |
| Balanced Acc | 0.9125 |
| Macro-Sensitivity | 0.9125 |
| Macro-Specificity | 0.9564 |
| Macro-Precision | 0.9127 |
| AUC-PR (Macro) | 0.9713 |
| Cohen's Kappa | 0.8694 |
| MCC | 0.8696 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 4818.1 (0.21 ms/张) |
| End-to-End FPS | 271.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8942 | 0.9429 | 0.8986 | 0.8964 | 0.9404 | 0.9775 | 0.9629 | 208 |
| Malignant_lesions | 0.8693 | 0.9575 | 0.9000 | 0.8844 | 0.9433 | 0.9793 | 0.9563 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9688 | 0.9397 | 0.9565 | 0.9867 | 0.9975 | 0.9947 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=186 FP=21 TN=347 FN=22
- **Malignant_lesions**: TP=153 FP=17 TN=383 FN=23
- **body_of_stomach_normal**: TP=187 FP=12 TN=372 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 186 | 16 | 6 |
| Malignant_lesions | 17 | 153 | 6 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8986    0.8942    0.8964       208
     Malignant_lesions     0.9000    0.8693    0.8844       176
body_of_stomach_normal     0.9397    0.9740    0.9565       192

              accuracy                         0.9132       576
             macro avg     0.9127    0.9125    0.9124       576
          weighted avg     0.9127    0.9132    0.9128       576

```