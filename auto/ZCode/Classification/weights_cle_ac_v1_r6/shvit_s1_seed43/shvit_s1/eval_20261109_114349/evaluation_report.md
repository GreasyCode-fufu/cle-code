# shvit_s1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/shvit_s1_seed43/shvit_s1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9425 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9918 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9425 |
| Balanced Acc | 0.9418 |
| Macro-Sensitivity | 0.9418 |
| Macro-Specificity | 0.9710 |
| Macro-Precision | 0.9439 |
| AUC-PR (Macro) | 0.9851 |
| Cohen's Kappa | 0.9138 |
| MCC | 0.9141 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 245.71M |
| FLOPs | 491.41M |
| Params | 6.01M |
| Pure-Inference FPS | 4902.9 (0.20 ms/张) |
| End-to-End FPS | 216.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9565 | 0.9242 | 0.9308 | 0.9644 | 0.9866 | 0.9794 | 208 |
| Malignant_lesions | 0.9034 | 0.9825 | 0.9578 | 0.9298 | 0.9585 | 0.9913 | 0.9805 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9740 | 0.9497 | 0.9668 | 0.9920 | 0.9975 | 0.9952 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=16 TN=352 FN=13
- **Malignant_lesions**: TP=159 FP=7 TN=393 FN=17
- **body_of_stomach_normal**: TP=189 FP=10 TN=374 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 7 | 6 |
| Malignant_lesions | 13 | 159 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9242    0.9375    0.9308       208
     Malignant_lesions     0.9578    0.9034    0.9298       176
body_of_stomach_normal     0.9497    0.9844    0.9668       192

              accuracy                         0.9427       576
             macro avg     0.9439    0.9418    0.9425       576
          weighted avg     0.9430    0.9427    0.9425       576

```