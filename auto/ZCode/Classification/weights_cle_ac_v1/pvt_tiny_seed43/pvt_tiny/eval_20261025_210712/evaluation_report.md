# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/pvt_tiny_seed43/pvt_tiny/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9394 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9921 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9394 |
| Balanced Acc | 0.9380 |
| Macro-Sensitivity | 0.9380 |
| Macro-Specificity | 0.9691 |
| Macro-Precision | 0.9416 |
| AUC-PR (Macro) | 0.9847 |
| Cohen's Kappa | 0.9085 |
| MCC | 0.9089 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 690.0 (1.45 ms/张) |
| End-to-End FPS | 203.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9402 | 0.8995 | 0.9227 | 0.9692 | 0.9883 | 0.9811 | 208 |
| Malignant_lesions | 0.9034 | 0.9775 | 0.9464 | 0.9244 | 0.9583 | 0.9905 | 0.9775 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9896 | 0.9788 | 0.9711 | 0.9819 | 0.9975 | 0.9956 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=22 TN=346 FN=11
- **Malignant_lesions**: TP=159 FP=9 TN=391 FN=17
- **body_of_stomach_normal**: TP=185 FP=4 TN=380 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 15 | 159 | 2 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8995    0.9471    0.9227       208
     Malignant_lesions     0.9464    0.9034    0.9244       176
body_of_stomach_normal     0.9788    0.9635    0.9711       192

              accuracy                         0.9392       576
             macro avg     0.9416    0.9380    0.9394       576
          weighted avg     0.9403    0.9392    0.9394       576

```