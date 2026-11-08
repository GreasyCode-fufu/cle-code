# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce4s_seed42/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8552 |
| Accuracy | 0.8644 |
| Macro-AUROC (OvR) | 0.9447 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8647 |
| Balanced Acc | 0.8521 |
| Macro-Sensitivity | 0.8521 |
| Macro-Specificity | 0.9212 |
| Macro-Precision | 0.8589 |
| AUC-PR (Macro) | 0.9006 |
| Cohen's Kappa | 0.7678 |
| MCC | 0.7679 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1725.0 (0.58 ms/张) |
| End-to-End FPS | 159.6 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8939 | 0.8654 | 0.8939 | 0.8939 | 0.8654 | 0.9321 | 0.9507 | 66 |
| malignant | 0.8125 | 0.9186 | 0.7879 | 0.8000 | 0.9294 | 0.9368 | 0.8319 | 32 |
| normal | 0.8500 | 0.9796 | 0.8947 | 0.8718 | 0.9697 | 0.9653 | 0.9191 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=7 TN=45 FN=7
- **malignant**: TP=26 FP=7 TN=79 FN=6
- **normal**: TP=17 FP=2 TN=96 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 7 | 0 |
| malignant | 4 | 26 | 2 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8939    0.8939    0.8939        66
   malignant     0.7879    0.8125    0.8000        32
      normal     0.8947    0.8500    0.8718        20

    accuracy                         0.8644       118
   macro avg     0.8589    0.8521    0.8552       118
weighted avg     0.8653    0.8644    0.8647       118

```