# mbv3_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce16_seed45/mbv3_dce16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9589 |
| Accuracy | 0.9672 |
| Macro-AUROC (OvR) | 0.9960 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9674 |
| Balanced Acc | 0.9635 |
| Macro-Sensitivity | 0.9635 |
| Macro-Specificity | 0.9844 |
| Macro-Precision | 0.9547 |
| AUC-PR (Macro) | 0.9878 |
| Cohen's Kappa | 0.9481 |
| MCC | 0.9483 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 3704.9 (0.27 ms/张) |
| End-to-End FPS | 22.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9538 | 0.9828 | 0.9789 | 0.9662 | 0.9620 | 0.9959 | 0.9951 | 195 |
| Malignant_lesions | 0.9367 | 0.9741 | 0.8916 | 0.9136 | 0.9855 | 0.9920 | 0.9682 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=186 FP=4 TN=228 FN=9
- **Malignant_lesions**: TP=74 FP=9 TN=339 FN=5
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 186 | 9 | 0 |
| Malignant_lesions | 4 | 74 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9789    0.9538    0.9662       195
     Malignant_lesions     0.8916    0.9367    0.9136        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9672       427
             macro avg     0.9547    0.9635    0.9589       427
          weighted avg     0.9680    0.9672    0.9674       427

```