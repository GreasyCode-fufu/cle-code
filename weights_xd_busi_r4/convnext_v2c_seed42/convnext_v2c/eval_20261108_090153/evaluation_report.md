# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r4/convnext_v2c_seed42/convnext_v2c/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8332 |
| Accuracy | 0.8475 |
| Macro-AUROC (OvR) | 0.9366 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8490 |
| Balanced Acc | 0.8528 |
| Macro-Sensitivity | 0.8528 |
| Macro-Specificity | 0.9204 |
| Macro-Precision | 0.8186 |
| AUC-PR (Macro) | 0.8727 |
| Cohen's Kappa | 0.7467 |
| MCC | 0.7493 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 708.8 (1.41 ms/张) |
| End-to-End FPS | 154.6 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8333 | 0.9038 | 0.9167 | 0.8730 | 0.8103 | 0.9164 | 0.9476 | 66 |
| malignant | 0.8750 | 0.9186 | 0.8000 | 0.8358 | 0.9518 | 0.9393 | 0.8581 | 32 |
| normal | 0.8500 | 0.9388 | 0.7391 | 0.7907 | 0.9684 | 0.9541 | 0.8125 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=55 FP=5 TN=47 FN=11
- **malignant**: TP=28 FP=7 TN=79 FN=4
- **normal**: TP=17 FP=6 TN=92 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 55 | 7 | 4 |
| malignant | 2 | 28 | 2 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9167    0.8333    0.8730        66
   malignant     0.8000    0.8750    0.8358        32
      normal     0.7391    0.8500    0.7907        20

    accuracy                         0.8475       118
   macro avg     0.8186    0.8528    0.8332       118
weighted avg     0.8549    0.8475    0.8490       118

```