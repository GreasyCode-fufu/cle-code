# convnext_v4a 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4a_seed43/convnext_v4a/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9752 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9970 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9789 |
| Balanced Acc | 0.9711 |
| Macro-Sensitivity | 0.9711 |
| Macro-Specificity | 0.9880 |
| Macro-Precision | 0.9796 |
| AUC-PR (Macro) | 0.9920 |
| Cohen's Kappa | 0.9664 |
| MCC | 0.9665 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 2.55G |
| FLOPs | 5.09G |
| Params | 17.62M |
| Pure-Inference FPS | 266.0 (3.76 ms/张) |
| End-to-End FPS | 142.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9897 | 0.9698 | 0.9650 | 0.9772 | 0.9912 | 0.9966 | 0.9961 | 195 |
| Malignant_lesions | 0.9367 | 0.9943 | 0.9737 | 0.9548 | 0.9858 | 0.9944 | 0.9800 | 79 |
| body_of_stomach_normal | 0.9869 | 1.0000 | 1.0000 | 0.9934 | 0.9928 | 0.9999 | 0.9997 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=7 TN=225 FN=2
- **Malignant_lesions**: TP=74 FP=2 TN=346 FN=5
- **body_of_stomach_normal**: TP=151 FP=0 TN=274 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 2 | 0 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 2 | 0 | 151 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9650    0.9897    0.9772       195
     Malignant_lesions     0.9737    0.9367    0.9548        79
body_of_stomach_normal     1.0000    0.9869    0.9934       153

              accuracy                         0.9789       427
             macro avg     0.9796    0.9711    0.9752       427
          weighted avg     0.9791    0.9789    0.9789       427

```