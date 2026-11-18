# lsnet_t 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/lsnet_t_seed46/lsnet_t/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8928 |
| Accuracy | 0.8958 |
| Macro-AUROC (OvR) | 0.9855 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8940 |
| Balanced Acc | 0.8905 |
| Macro-Sensitivity | 0.8905 |
| Macro-Specificity | 0.9467 |
| Macro-Precision | 0.9057 |
| AUC-PR (Macro) | 0.9720 |
| Cohen's Kappa | 0.8426 |
| MCC | 0.8470 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 320.32M |
| FLOPs | 640.63M |
| Params | 11.02M |
| Pure-Inference FPS | 4856.8 (0.21 ms/张) |
| End-to-End FPS | 218.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.8940 | 0.8347 | 0.8874 | 0.9676 | 0.9795 | 0.9678 | 208 |
| Malignant_lesions | 0.7557 | 0.9850 | 0.9568 | 0.8444 | 0.9016 | 0.9815 | 0.9558 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9609 | 0.9254 | 0.9466 | 0.9840 | 0.9956 | 0.9924 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=39 TN=329 FN=11
- **Malignant_lesions**: TP=133 FP=6 TN=394 FN=43
- **body_of_stomach_normal**: TP=186 FP=15 TN=369 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 6 | 5 |
| Malignant_lesions | 33 | 133 | 10 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8347    0.9471    0.8874       208
     Malignant_lesions     0.9568    0.7557    0.8444       176
body_of_stomach_normal     0.9254    0.9688    0.9466       192

              accuracy                         0.8958       576
             macro avg     0.9057    0.8905    0.8928       576
          weighted avg     0.9023    0.8958    0.8940       576

```