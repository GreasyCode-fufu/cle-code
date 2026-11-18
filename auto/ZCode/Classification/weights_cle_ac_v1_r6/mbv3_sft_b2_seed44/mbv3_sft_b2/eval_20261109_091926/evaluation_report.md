# mbv3_sft_b2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b2_seed44/mbv3_sft_b2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9167 |
| Accuracy | 0.9184 |
| Macro-AUROC (OvR) | 0.9899 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9174 |
| Balanced Acc | 0.9134 |
| Macro-Sensitivity | 0.9134 |
| Macro-Specificity | 0.9579 |
| Macro-Precision | 0.9297 |
| AUC-PR (Macro) | 0.9810 |
| Cohen's Kappa | 0.8767 |
| MCC | 0.8810 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2425.3 (0.41 ms/张) |
| End-to-End FPS | 151.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9022 | 0.8494 | 0.9083 | 0.9852 | 0.9853 | 0.9732 | 208 |
| Malignant_lesions | 0.7955 | 0.9950 | 0.9859 | 0.8805 | 0.9171 | 0.9867 | 0.9745 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9766 | 0.9538 | 0.9612 | 0.9843 | 0.9976 | 0.9953 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=36 TN=332 FN=5
- **Malignant_lesions**: TP=140 FP=2 TN=398 FN=36
- **body_of_stomach_normal**: TP=186 FP=9 TN=375 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 2 | 3 |
| Malignant_lesions | 30 | 140 | 6 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8494    0.9760    0.9083       208
     Malignant_lesions     0.9859    0.7955    0.8805       176
body_of_stomach_normal     0.9538    0.9688    0.9612       192

              accuracy                         0.9184       576
             macro avg     0.9297    0.9134    0.9167       576
          weighted avg     0.9259    0.9184    0.9174       576

```