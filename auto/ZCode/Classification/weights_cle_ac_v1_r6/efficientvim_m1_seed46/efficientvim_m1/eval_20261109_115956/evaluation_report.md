# efficientvim_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/efficientvim_m1_seed46/efficientvim_m1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9408 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9910 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9409 |
| Balanced Acc | 0.9392 |
| Macro-Sensitivity | 0.9392 |
| Macro-Specificity | 0.9699 |
| Macro-Precision | 0.9438 |
| AUC-PR (Macro) | 0.9831 |
| Cohen's Kappa | 0.9111 |
| MCC | 0.9117 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 234.46M |
| FLOPs | 468.92M |
| Params | 5.71M |
| Pure-Inference FPS | 6522.3 (0.15 ms/张) |
| End-to-End FPS | 265.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9429 | 0.9045 | 0.9299 | 0.9747 | 0.9861 | 0.9786 | 208 |
| Malignant_lesions | 0.8920 | 0.9850 | 0.9632 | 0.9263 | 0.9540 | 0.9897 | 0.9760 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9818 | 0.9637 | 0.9662 | 0.9843 | 0.9972 | 0.9948 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=21 TN=347 FN=9
- **Malignant_lesions**: TP=157 FP=6 TN=394 FN=19
- **body_of_stomach_normal**: TP=186 FP=7 TN=377 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 6 | 3 |
| Malignant_lesions | 15 | 157 | 4 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9045    0.9567    0.9299       208
     Malignant_lesions     0.9632    0.8920    0.9263       176
body_of_stomach_normal     0.9637    0.9688    0.9662       192

              accuracy                         0.9410       576
             macro avg     0.9438    0.9392    0.9408       576
          weighted avg     0.9422    0.9410    0.9409       576

```