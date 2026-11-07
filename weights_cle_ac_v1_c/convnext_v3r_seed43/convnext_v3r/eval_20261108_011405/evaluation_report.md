# convnext_v3r 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v3r_seed43/convnext_v3r/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9480 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9952 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9480 |
| Balanced Acc | 0.9463 |
| Macro-Sensitivity | 0.9463 |
| Macro-Specificity | 0.9734 |
| Macro-Precision | 0.9509 |
| AUC-PR (Macro) | 0.9910 |
| Cohen's Kappa | 0.9216 |
| MCC | 0.9222 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.65G |
| FLOPs | 9.29G |
| Params | 28.68M |
| Pure-Inference FPS | 1409.1 (0.71 ms/张) |
| End-to-End FPS | 265.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9429 | 0.9050 | 0.9324 | 0.9775 | 0.9934 | 0.9896 | 208 |
| Malignant_lesions | 0.9034 | 0.9850 | 0.9636 | 0.9326 | 0.9586 | 0.9946 | 0.9865 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9922 | 0.9842 | 0.9791 | 0.9870 | 0.9977 | 0.9968 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=21 TN=347 FN=8
- **Malignant_lesions**: TP=159 FP=6 TN=394 FN=17
- **body_of_stomach_normal**: TP=187 FP=3 TN=381 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 16 | 159 | 1 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9050    0.9615    0.9324       208
     Malignant_lesions     0.9636    0.9034    0.9326       176
body_of_stomach_normal     0.9842    0.9740    0.9791       192

              accuracy                         0.9479       576
             macro avg     0.9509    0.9463    0.9480       576
          weighted avg     0.9493    0.9479    0.9480       576

```