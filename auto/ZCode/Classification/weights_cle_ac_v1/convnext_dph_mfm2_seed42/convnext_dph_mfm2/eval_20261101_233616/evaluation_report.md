# convnext_dph_mfm2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_mfm2_seed42/convnext_dph_mfm2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9467 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9474 |
| Balanced Acc | 0.9457 |
| Macro-Sensitivity | 0.9457 |
| Macro-Specificity | 0.9736 |
| Macro-Precision | 0.9496 |
| AUC-PR (Macro) | 0.9934 |
| Cohen's Kappa | 0.9216 |
| MCC | 0.9224 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 856.3 (1.17 ms/张) |
| End-to-End FPS | 221.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9620 | 0.9346 | 0.9479 | 0.9779 | 0.9960 | 0.9936 | 208 |
| Malignant_lesions | 0.8807 | 0.9875 | 0.9688 | 0.9226 | 0.9495 | 0.9949 | 0.9876 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9714 | 0.9455 | 0.9695 | 0.9973 | 0.9995 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=14 TN=354 FN=8
- **Malignant_lesions**: TP=155 FP=5 TN=395 FN=21
- **body_of_stomach_normal**: TP=191 FP=11 TN=373 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 5 | 3 |
| Malignant_lesions | 13 | 155 | 8 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9346    0.9615    0.9479       208
     Malignant_lesions     0.9688    0.8807    0.9226       176
body_of_stomach_normal     0.9455    0.9948    0.9695       192

              accuracy                         0.9479       576
             macro avg     0.9496    0.9457    0.9467       576
          weighted avg     0.9487    0.9479    0.9474       576

```