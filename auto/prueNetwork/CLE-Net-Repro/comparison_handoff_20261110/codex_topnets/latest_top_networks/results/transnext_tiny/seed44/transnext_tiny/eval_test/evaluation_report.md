# transnext_tiny 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/transnext_tiny/seed44/transnext_tiny/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9422 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9937 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9423 |
| Balanced Acc | 0.9436 |
| Macro-Sensitivity | 0.9436 |
| Macro-Specificity | 0.9716 |
| Macro-Precision | 0.9418 |
| AUC-PR (Macro) | 0.9871 |
| Cohen's Kappa | 0.9140 |
| MCC | 0.9145 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 5.53G |
| FLOPs | 11.07G |
| Params | 27.63M |
| Pure-Inference FPS | 114.7 (8.72 ms/张) |
| End-to-End FPS | 14.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8990 | 0.9783 | 0.9590 | 0.9280 | 0.9449 | 0.9915 | 0.9869 | 208 |
| Malignant_lesions | 0.9318 | 0.9600 | 0.9111 | 0.9213 | 0.9697 | 0.9896 | 0.9745 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9766 | 0.9552 | 0.9771 | 1.0000 | 1.0000 | 0.9999 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=8 TN=360 FN=21
- **Malignant_lesions**: TP=164 FP=16 TN=384 FN=12
- **body_of_stomach_normal**: TP=192 FP=9 TN=375 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 16 | 5 |
| Malignant_lesions | 8 | 164 | 4 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9590    0.8990    0.9280       208
     Malignant_lesions     0.9111    0.9318    0.9213       176
body_of_stomach_normal     0.9552    1.0000    0.9771       192

              accuracy                         0.9427       576
             macro avg     0.9418    0.9436    0.9422       576
          weighted avg     0.9431    0.9427    0.9423       576

```