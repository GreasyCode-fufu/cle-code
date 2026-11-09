# mbv3_l2sp_a01 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_l2sp_a01_seed46/mbv3_l2sp_a01/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9710 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9767 |
| Balanced Acc | 0.9715 |
| Macro-Sensitivity | 0.9715 |
| Macro-Specificity | 0.9880 |
| Macro-Precision | 0.9707 |
| AUC-PR (Macro) | 0.9903 |
| Cohen's Kappa | 0.9628 |
| MCC | 0.9628 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2898.7 (0.34 ms/张) |
| End-to-End FPS | 199.7 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9784 | 0.9746 | 0.9796 | 0.9870 | 0.9968 | 0.9964 | 195 |
| Malignant_lesions | 0.9494 | 0.9856 | 0.9375 | 0.9434 | 0.9885 | 0.9937 | 0.9745 | 79 |
| body_of_stomach_normal | 0.9804 | 1.0000 | 1.0000 | 0.9901 | 0.9892 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=5 TN=227 FN=3
- **Malignant_lesions**: TP=75 FP=5 TN=343 FN=4
- **body_of_stomach_normal**: TP=150 FP=0 TN=274 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 3 | 0 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 1 | 2 | 150 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9746    0.9846    0.9796       195
     Malignant_lesions     0.9375    0.9494    0.9434        79
body_of_stomach_normal     1.0000    0.9804    0.9901       153

              accuracy                         0.9766       427
             macro avg     0.9707    0.9715    0.9710       427
          weighted avg     0.9768    0.9766    0.9767       427

```