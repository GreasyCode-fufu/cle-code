# mbv3_r6_star1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star1_seed44/mbv3_r6_star1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9142 |
| Accuracy | 0.9149 |
| Macro-AUROC (OvR) | 0.9888 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9147 |
| Balanced Acc | 0.9106 |
| Macro-Sensitivity | 0.9106 |
| Macro-Specificity | 0.9562 |
| Macro-Precision | 0.9250 |
| AUC-PR (Macro) | 0.9770 |
| Cohen's Kappa | 0.8716 |
| MCC | 0.8751 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.18M |
| FLOPs | 464.35M |
| Params | 3.47M |
| Pure-Inference FPS | 3641.8 (0.27 ms/张) |
| End-to-End FPS | 225.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.8940 | 0.8368 | 0.8949 | 0.9763 | 0.9828 | 0.9700 | 208 |
| Malignant_lesions | 0.8068 | 0.9850 | 0.9595 | 0.8765 | 0.9206 | 0.9852 | 0.9639 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9896 | 0.9788 | 0.9711 | 0.9819 | 0.9985 | 0.9970 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=39 TN=329 FN=8
- **Malignant_lesions**: TP=142 FP=6 TN=394 FN=34
- **body_of_stomach_normal**: TP=185 FP=4 TN=380 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 32 | 142 | 2 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8368    0.9615    0.8949       208
     Malignant_lesions     0.9595    0.8068    0.8765       176
body_of_stomach_normal     0.9788    0.9635    0.9711       192

              accuracy                         0.9149       576
             macro avg     0.9250    0.9106    0.9142       576
          weighted avg     0.9216    0.9149    0.9147       576

```