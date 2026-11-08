# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce16_seed44/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8546 |
| Accuracy | 0.8729 |
| Macro-AUROC (OvR) | 0.9450 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8729 |
| Balanced Acc | 0.8572 |
| Macro-Sensitivity | 0.8572 |
| Macro-Specificity | 0.9286 |
| Macro-Precision | 0.8524 |
| AUC-PR (Macro) | 0.9081 |
| Cohen's Kappa | 0.7830 |
| MCC | 0.7831 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 1714.3 (0.58 ms/张) |
| End-to-End FPS | 160.2 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9091 | 0.8846 | 0.9091 | 0.9091 | 0.8846 | 0.9385 | 0.9429 | 66 |
| malignant | 0.8125 | 0.9419 | 0.8387 | 0.8254 | 0.9310 | 0.9484 | 0.8899 | 32 |
| normal | 0.8500 | 0.9592 | 0.8095 | 0.8293 | 0.9691 | 0.9480 | 0.8914 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=6 TN=46 FN=6
- **malignant**: TP=26 FP=5 TN=81 FN=6
- **normal**: TP=17 FP=4 TN=94 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 5 | 1 |
| malignant | 3 | 26 | 3 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9091    0.9091    0.9091        66
   malignant     0.8387    0.8125    0.8254        32
      normal     0.8095    0.8500    0.8293        20

    accuracy                         0.8729       118
   macro avg     0.8524    0.8572    0.8546       118
weighted avg     0.8731    0.8729    0.8729       118

```