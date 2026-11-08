# shufflenet_v2_x1_0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/shufflenet_v2_x1_0_seed42/shufflenet_v2_x1_0/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9665 |
| Accuracy | 0.9742 |
| Macro-AUROC (OvR) | 0.9942 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9739 |
| Balanced Acc | 0.9611 |
| Macro-Sensitivity | 0.9611 |
| Macro-Specificity | 0.9859 |
| Macro-Precision | 0.9726 |
| AUC-PR (Macro) | 0.9837 |
| Cohen's Kappa | 0.9588 |
| MCC | 0.9590 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 151.69M |
| FLOPs | 303.38M |
| Params | 1.26M |
| Pure-Inference FPS | 5265.9 (0.19 ms/张) |
| End-to-End FPS | 83.7 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9698 | 0.9648 | 0.9746 | 0.9868 | 0.9939 | 0.9922 | 195 |
| Malignant_lesions | 0.8987 | 0.9914 | 0.9595 | 0.9281 | 0.9773 | 0.9888 | 0.9589 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=7 TN=225 FN=3
- **Malignant_lesions**: TP=71 FP=3 TN=345 FN=8
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 3 | 0 |
| Malignant_lesions | 7 | 71 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9648    0.9846    0.9746       195
     Malignant_lesions     0.9595    0.8987    0.9281        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9742       427
             macro avg     0.9726    0.9611    0.9665       427
          weighted avg     0.9741    0.9742    0.9739       427

```