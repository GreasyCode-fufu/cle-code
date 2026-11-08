# starnet_s2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_seed46/starnet_s2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9628 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9946 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9693 |
| Balanced Acc | 0.9577 |
| Macro-Sensitivity | 0.9577 |
| Macro-Specificity | 0.9832 |
| Macro-Precision | 0.9687 |
| AUC-PR (Macro) | 0.9821 |
| Cohen's Kappa | 0.9514 |
| MCC | 0.9515 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.29M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 4248.0 (0.24 ms/张) |
| End-to-End FPS | 245.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9655 | 0.9596 | 0.9669 | 0.9782 | 0.9919 | 0.9933 | 195 |
| Malignant_lesions | 0.8987 | 0.9914 | 0.9595 | 0.9281 | 0.9773 | 0.9919 | 0.9529 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=8 TN=224 FN=5
- **Malignant_lesions**: TP=71 FP=3 TN=345 FN=8
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 3 | 2 |
| Malignant_lesions | 8 | 71 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9596    0.9744    0.9669       195
     Malignant_lesions     0.9595    0.8987    0.9281        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9696       427
             macro avg     0.9687    0.9577    0.9628       427
          weighted avg     0.9694    0.9696    0.9693       427

```