# mobilenetv4_conv_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mobilenetv4_conv_small_seed44/mobilenetv4_conv_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8935 |
| Accuracy | 0.8941 |
| Macro-AUROC (OvR) | 0.9803 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8937 |
| Balanced Acc | 0.8926 |
| Macro-Sensitivity | 0.8926 |
| Macro-Specificity | 0.9464 |
| Macro-Precision | 0.8955 |
| AUC-PR (Macro) | 0.9613 |
| Cohen's Kappa | 0.8405 |
| MCC | 0.8410 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 184.77M |
| FLOPs | 369.55M |
| Params | 2.47M |
| Pure-Inference FPS | 7217.3 (0.14 ms/张) |
| End-to-End FPS | 127.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8846 | 0.9130 | 0.8519 | 0.8679 | 0.9333 | 0.9702 | 0.9526 | 208 |
| Malignant_lesions | 0.8295 | 0.9575 | 0.8957 | 0.8614 | 0.9274 | 0.9743 | 0.9381 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9688 | 0.9391 | 0.9512 | 0.9815 | 0.9963 | 0.9933 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=184 FP=32 TN=336 FN=24
- **Malignant_lesions**: TP=146 FP=17 TN=383 FN=30
- **body_of_stomach_normal**: TP=185 FP=12 TN=372 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 184 | 16 | 8 |
| Malignant_lesions | 26 | 146 | 4 |
| body_of_stomach_normal | 6 | 1 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8519    0.8846    0.8679       208
     Malignant_lesions     0.8957    0.8295    0.8614       176
body_of_stomach_normal     0.9391    0.9635    0.9512       192

              accuracy                         0.8941       576
             macro avg     0.8955    0.8926    0.8935       576
          weighted avg     0.8943    0.8941    0.8937       576

```