# mbv3_sft_b2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b2_seed43/mbv3_sft_b2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9620 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9950 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9695 |
| Balanced Acc | 0.9622 |
| Macro-Sensitivity | 0.9622 |
| Macro-Specificity | 0.9846 |
| Macro-Precision | 0.9618 |
| AUC-PR (Macro) | 0.9830 |
| Cohen's Kappa | 0.9516 |
| MCC | 0.9516 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2082.8 (0.48 ms/张) |
| End-to-End FPS | 104.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9784 | 0.9742 | 0.9717 | 0.9742 | 0.9950 | 0.9942 | 195 |
| Malignant_lesions | 0.9241 | 0.9828 | 0.9241 | 0.9241 | 0.9828 | 0.9901 | 0.9550 | 79 |
| body_of_stomach_normal | 0.9935 | 0.9927 | 0.9870 | 0.9902 | 0.9963 | 0.9999 | 0.9999 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=5 TN=227 FN=6
- **Malignant_lesions**: TP=73 FP=6 TN=342 FN=6
- **body_of_stomach_normal**: TP=152 FP=2 TN=272 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 6 | 0 |
| Malignant_lesions | 4 | 73 | 2 |
| body_of_stomach_normal | 1 | 0 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9742    0.9692    0.9717       195
     Malignant_lesions     0.9241    0.9241    0.9241        79
body_of_stomach_normal     0.9870    0.9935    0.9902       153

              accuracy                         0.9696       427
             macro avg     0.9618    0.9622    0.9620       427
          weighted avg     0.9695    0.9696    0.9695       427

```