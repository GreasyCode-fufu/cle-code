# mbv3_r6_s2_lsside 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_lsside_seed43/mbv3_r6_s2_lsside/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9387 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9898 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9390 |
| Balanced Acc | 0.9375 |
| Macro-Sensitivity | 0.9375 |
| Macro-Specificity | 0.9691 |
| Macro-Precision | 0.9409 |
| AUC-PR (Macro) | 0.9801 |
| Cohen's Kappa | 0.9085 |
| MCC | 0.9090 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 235.94M |
| FLOPs | 471.88M |
| Params | 3.27M |
| Pure-Inference FPS | 2271.4 (0.44 ms/张) |
| End-to-End FPS | 150.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9429 | 0.9032 | 0.9224 | 0.9666 | 0.9845 | 0.9714 | 208 |
| Malignant_lesions | 0.8807 | 0.9775 | 0.9451 | 0.9118 | 0.9490 | 0.9854 | 0.9703 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=21 TN=347 FN=12
- **Malignant_lesions**: TP=155 FP=9 TN=391 FN=21
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 9 | 3 |
| Malignant_lesions | 19 | 155 | 2 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9032    0.9423    0.9224       208
     Malignant_lesions     0.9451    0.8807    0.9118       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9392       576
             macro avg     0.9409    0.9375    0.9387       576
          weighted avg     0.9397    0.9392    0.9390       576

```