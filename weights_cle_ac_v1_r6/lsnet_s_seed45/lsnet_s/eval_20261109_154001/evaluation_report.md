# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/lsnet_s_seed45/lsnet_s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9100 |
| Accuracy | 0.9115 |
| Macro-AUROC (OvR) | 0.9875 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9108 |
| Balanced Acc | 0.9082 |
| Macro-Sensitivity | 0.9082 |
| Macro-Specificity | 0.9549 |
| Macro-Precision | 0.9156 |
| AUC-PR (Macro) | 0.9767 |
| Cohen's Kappa | 0.8665 |
| MCC | 0.8681 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 3415.8 (0.29 ms/张) |
| End-to-End FPS | 223.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9212 | 0.8711 | 0.9053 | 0.9658 | 0.9857 | 0.9758 | 208 |
| Malignant_lesions | 0.8239 | 0.9775 | 0.9416 | 0.8788 | 0.9265 | 0.9829 | 0.9667 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9661 | 0.9340 | 0.9460 | 0.9789 | 0.9940 | 0.9877 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=29 TN=339 FN=12
- **Malignant_lesions**: TP=145 FP=9 TN=391 FN=31
- **body_of_stomach_normal**: TP=184 FP=13 TN=371 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 8 | 4 |
| Malignant_lesions | 22 | 145 | 9 |
| body_of_stomach_normal | 7 | 1 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8711    0.9423    0.9053       208
     Malignant_lesions     0.9416    0.8239    0.8788       176
body_of_stomach_normal     0.9340    0.9583    0.9460       192

              accuracy                         0.9115       576
             macro avg     0.9156    0.9082    0.9100       576
          weighted avg     0.9136    0.9115    0.9108       576

```