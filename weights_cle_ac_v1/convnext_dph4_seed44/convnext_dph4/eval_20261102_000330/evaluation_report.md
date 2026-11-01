# convnext_dph4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph4_seed44/convnext_dph4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9775 |
| Accuracy | 0.9813 |
| Macro-AUROC (OvR) | 0.9980 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9813 |
| Balanced Acc | 0.9813 |
| Macro-Sensitivity | 0.9813 |
| Macro-Specificity | 0.9911 |
| Macro-Precision | 0.9740 |
| AUC-PR (Macro) | 0.9928 |
| Cohen's Kappa | 0.9703 |
| MCC | 0.9704 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 452.5 (2.21 ms/张) |
| End-to-End FPS | 184.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9914 | 0.9895 | 0.9793 | 0.9746 | 0.9977 | 0.9975 | 195 |
| Malignant_lesions | 0.9747 | 0.9856 | 0.9390 | 0.9565 | 0.9942 | 0.9963 | 0.9810 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=2 TN=230 FN=6
- **Malignant_lesions**: TP=77 FP=5 TN=343 FN=2
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 5 | 1 |
| Malignant_lesions | 2 | 77 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9895    0.9692    0.9793       195
     Malignant_lesions     0.9390    0.9747    0.9565        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9813       427
             macro avg     0.9740    0.9813    0.9775       427
          weighted avg     0.9816    0.9813    0.9813       427

```