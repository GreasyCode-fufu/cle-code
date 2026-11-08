# shufflenet_v2_x1_0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/shufflenet_v2_x1_0_seed46/shufflenet_v2_x1_0/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9218 |
| Accuracy | 0.9219 |
| Macro-AUROC (OvR) | 0.9858 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9220 |
| Balanced Acc | 0.9191 |
| Macro-Sensitivity | 0.9191 |
| Macro-Specificity | 0.9600 |
| Macro-Precision | 0.9276 |
| AUC-PR (Macro) | 0.9737 |
| Cohen's Kappa | 0.8822 |
| MCC | 0.8838 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 151.69M |
| FLOPs | 303.38M |
| Params | 1.26M |
| Pure-Inference FPS | 8244.2 (0.12 ms/张) |
| End-to-End FPS | 189.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9130 | 0.8609 | 0.9041 | 0.9711 | 0.9820 | 0.9668 | 208 |
| Malignant_lesions | 0.8523 | 0.9775 | 0.9434 | 0.8955 | 0.9376 | 0.9800 | 0.9607 | 176 |
| body_of_stomach_normal | 0.9531 | 0.9896 | 0.9786 | 0.9657 | 0.9769 | 0.9953 | 0.9935 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=32 TN=336 FN=10
- **Malignant_lesions**: TP=150 FP=9 TN=391 FN=26
- **body_of_stomach_normal**: TP=183 FP=4 TN=380 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 8 | 2 |
| Malignant_lesions | 24 | 150 | 2 |
| body_of_stomach_normal | 8 | 1 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8609    0.9519    0.9041       208
     Malignant_lesions     0.9434    0.8523    0.8955       176
body_of_stomach_normal     0.9786    0.9531    0.9657       192

              accuracy                         0.9219       576
             macro avg     0.9276    0.9191    0.9218       576
          weighted avg     0.9253    0.9219    0.9220       576

```