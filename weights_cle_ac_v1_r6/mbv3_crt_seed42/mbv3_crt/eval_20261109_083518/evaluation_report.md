# mbv3_crt 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_crt_seed42/mbv3_crt/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9482 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9941 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9480 |
| Balanced Acc | 0.9466 |
| Macro-Sensitivity | 0.9466 |
| Macro-Specificity | 0.9734 |
| Macro-Precision | 0.9507 |
| AUC-PR (Macro) | 0.9885 |
| Cohen's Kappa | 0.9216 |
| MCC | 0.9221 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 2725.3 (0.37 ms/张) |
| End-to-End FPS | 7.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9457 | 0.9091 | 0.9346 | 0.9775 | 0.9916 | 0.9862 | 208 |
| Malignant_lesions | 0.9148 | 0.9850 | 0.9641 | 0.9388 | 0.9633 | 0.9923 | 0.9821 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9896 | 0.9788 | 0.9711 | 0.9819 | 0.9985 | 0.9973 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=20 TN=348 FN=8
- **Malignant_lesions**: TP=161 FP=6 TN=394 FN=15
- **body_of_stomach_normal**: TP=185 FP=4 TN=380 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 13 | 161 | 2 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9091    0.9615    0.9346       208
     Malignant_lesions     0.9641    0.9148    0.9388       176
body_of_stomach_normal     0.9788    0.9635    0.9711       192

              accuracy                         0.9479       576
             macro avg     0.9507    0.9466    0.9482       576
          weighted avg     0.9491    0.9479    0.9480       576

```