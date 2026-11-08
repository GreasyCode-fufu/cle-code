# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz8_dce4m_seed44/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8197 |
| Accuracy | 0.8534 |
| Macro-AUROC (OvR) | 0.9244 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8525 |
| Balanced Acc | 0.8158 |
| Macro-Sensitivity | 0.8158 |
| Macro-Specificity | 0.9173 |
| Macro-Precision | 0.8244 |
| AUC-PR (Macro) | 0.8847 |
| Cohen's Kappa | 0.7472 |
| MCC | 0.7477 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1586.7 (0.63 ms/张) |
| End-to-End FPS | 12.6 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9231 | 0.8627 | 0.8955 | 0.9091 | 0.8980 | 0.9261 | 0.9090 | 65 |
| malignant | 0.7742 | 0.9412 | 0.8276 | 0.8000 | 0.9195 | 0.9268 | 0.8948 | 31 |
| normal | 0.7500 | 0.9479 | 0.7500 | 0.7500 | 0.9479 | 0.9203 | 0.8502 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=7 TN=44 FN=5
- **malignant**: TP=24 FP=5 TN=80 FN=7
- **normal**: TP=15 FP=5 TN=91 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 4 | 1 |
| malignant | 3 | 24 | 4 |
| normal | 4 | 1 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8955    0.9231    0.9091        65
   malignant     0.8276    0.7742    0.8000        31
      normal     0.7500    0.7500    0.7500        20

    accuracy                         0.8534       116
   macro avg     0.8244    0.8158    0.8197       116
weighted avg     0.8523    0.8534    0.8525       116

```