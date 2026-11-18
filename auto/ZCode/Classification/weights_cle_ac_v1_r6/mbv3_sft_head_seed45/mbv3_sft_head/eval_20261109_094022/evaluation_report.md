# mbv3_sft_head 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_head_seed45/mbv3_sft_head/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8680 |
| Accuracy | 0.8698 |
| Macro-AUROC (OvR) | 0.9745 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8682 |
| Balanced Acc | 0.8672 |
| Macro-Sensitivity | 0.8672 |
| Macro-Specificity | 0.9340 |
| Macro-Precision | 0.8773 |
| AUC-PR (Macro) | 0.9542 |
| Cohen's Kappa | 0.8037 |
| MCC | 0.8071 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3219.2 (0.31 ms/张) |
| End-to-End FPS | 135.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8606 | 0.9103 | 0.8443 | 0.8524 | 0.9203 | 0.9566 | 0.9309 | 208 |
| Malignant_lesions | 0.7670 | 0.9775 | 0.9375 | 0.8438 | 0.9051 | 0.9767 | 0.9514 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9141 | 0.8500 | 0.9078 | 0.9860 | 0.9900 | 0.9803 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=179 FP=33 TN=335 FN=29
- **Malignant_lesions**: TP=135 FP=9 TN=391 FN=41
- **body_of_stomach_normal**: TP=187 FP=33 TN=351 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 179 | 7 | 22 |
| Malignant_lesions | 30 | 135 | 11 |
| body_of_stomach_normal | 3 | 2 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8443    0.8606    0.8524       208
     Malignant_lesions     0.9375    0.7670    0.8438       176
body_of_stomach_normal     0.8500    0.9740    0.9078       192

              accuracy                         0.8698       576
             macro avg     0.8773    0.8672    0.8680       576
          weighted avg     0.8747    0.8698    0.8682       576

```