# convnext_epnet2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_epnet2_seed43/convnext_epnet2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9392 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9919 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9394 |
| Balanced Acc | 0.9377 |
| Macro-Sensitivity | 0.9377 |
| Macro-Specificity | 0.9691 |
| Macro-Precision | 0.9415 |
| AUC-PR (Macro) | 0.9852 |
| Cohen's Kappa | 0.9085 |
| MCC | 0.9090 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 28.57M |
| Pure-Inference FPS | 1706.4 (0.59 ms/张) |
| End-to-End FPS | 270.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9402 | 0.9000 | 0.9252 | 0.9719 | 0.9893 | 0.9824 | 208 |
| Malignant_lesions | 0.8977 | 0.9750 | 0.9405 | 0.9186 | 0.9559 | 0.9891 | 0.9775 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9922 | 0.9840 | 0.9737 | 0.9820 | 0.9972 | 0.9958 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=22 TN=346 FN=10
- **Malignant_lesions**: TP=158 FP=10 TN=390 FN=18
- **body_of_stomach_normal**: TP=185 FP=3 TN=381 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 9 | 1 |
| Malignant_lesions | 16 | 158 | 2 |
| body_of_stomach_normal | 6 | 1 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9000    0.9519    0.9252       208
     Malignant_lesions     0.9405    0.8977    0.9186       176
body_of_stomach_normal     0.9840    0.9635    0.9737       192

              accuracy                         0.9392       576
             macro avg     0.9415    0.9377    0.9392       576
          weighted avg     0.9404    0.9392    0.9394       576

```