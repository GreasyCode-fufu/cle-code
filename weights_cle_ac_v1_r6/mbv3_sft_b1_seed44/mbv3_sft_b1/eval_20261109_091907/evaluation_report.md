# mbv3_sft_b1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b1_seed44/mbv3_sft_b1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9455 |
| Accuracy | 0.9555 |
| Macro-AUROC (OvR) | 0.9945 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9556 |
| Balanced Acc | 0.9460 |
| Macro-Sensitivity | 0.9460 |
| Macro-Specificity | 0.9772 |
| Macro-Precision | 0.9450 |
| AUC-PR (Macro) | 0.9844 |
| Cohen's Kappa | 0.9293 |
| MCC | 0.9293 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1688.4 (0.59 ms/张) |
| End-to-End FPS | 107.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9612 | 0.9541 | 0.9565 | 0.9654 | 0.9936 | 0.9929 | 195 |
| Malignant_lesions | 0.8987 | 0.9741 | 0.8875 | 0.8931 | 0.9769 | 0.9905 | 0.9613 | 79 |
| body_of_stomach_normal | 0.9804 | 0.9964 | 0.9934 | 0.9868 | 0.9891 | 0.9995 | 0.9992 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=9 TN=223 FN=8
- **Malignant_lesions**: TP=71 FP=9 TN=339 FN=8
- **body_of_stomach_normal**: TP=150 FP=1 TN=273 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 8 | 0 |
| Malignant_lesions | 7 | 71 | 1 |
| body_of_stomach_normal | 2 | 1 | 150 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9541    0.9590    0.9565       195
     Malignant_lesions     0.8875    0.8987    0.8931        79
body_of_stomach_normal     0.9934    0.9804    0.9868       153

              accuracy                         0.9555       427
             macro avg     0.9450    0.9460    0.9455       427
          weighted avg     0.9558    0.9555    0.9556       427

```