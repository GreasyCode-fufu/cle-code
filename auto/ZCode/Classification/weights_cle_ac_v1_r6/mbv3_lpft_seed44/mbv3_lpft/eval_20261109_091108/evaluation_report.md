# mbv3_lpft 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_lpft_seed44/mbv3_lpft/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9627 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9965 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9697 |
| Balanced Acc | 0.9677 |
| Macro-Sensitivity | 0.9677 |
| Macro-Specificity | 0.9856 |
| Macro-Precision | 0.9583 |
| AUC-PR (Macro) | 0.9870 |
| Cohen's Kappa | 0.9518 |
| MCC | 0.9521 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 2410.0 (0.41 ms/张) |
| End-to-End FPS | 140.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9538 | 0.9871 | 0.9841 | 0.9688 | 0.9622 | 0.9962 | 0.9959 | 195 |
| Malignant_lesions | 0.9494 | 0.9770 | 0.9036 | 0.9259 | 0.9884 | 0.9932 | 0.9650 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=186 FP=3 TN=229 FN=9
- **Malignant_lesions**: TP=75 FP=8 TN=340 FN=4
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 186 | 8 | 1 |
| Malignant_lesions | 3 | 75 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9841    0.9538    0.9688       195
     Malignant_lesions     0.9036    0.9494    0.9259        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9696       427
             macro avg     0.9583    0.9677    0.9627       427
          weighted avg     0.9703    0.9696    0.9697       427

```