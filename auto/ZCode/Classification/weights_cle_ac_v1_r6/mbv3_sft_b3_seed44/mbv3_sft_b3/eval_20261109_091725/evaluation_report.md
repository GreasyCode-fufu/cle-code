# mbv3_sft_b3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b3_seed44/mbv3_sft_b3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9379 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9928 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9378 |
| Balanced Acc | 0.9354 |
| Macro-Sensitivity | 0.9354 |
| Macro-Specificity | 0.9680 |
| Macro-Precision | 0.9427 |
| AUC-PR (Macro) | 0.9866 |
| Cohen's Kappa | 0.9058 |
| MCC | 0.9071 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3070.8 (0.33 ms/张) |
| End-to-End FPS | 136.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9266 | 0.8816 | 0.9220 | 0.9799 | 0.9890 | 0.9827 | 208 |
| Malignant_lesions | 0.8920 | 0.9825 | 0.9573 | 0.9235 | 0.9539 | 0.9911 | 0.9801 | 176 |
| body_of_stomach_normal | 0.9479 | 0.9948 | 0.9891 | 0.9681 | 0.9745 | 0.9984 | 0.9969 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=27 TN=341 FN=7
- **Malignant_lesions**: TP=157 FP=7 TN=393 FN=19
- **body_of_stomach_normal**: TP=182 FP=2 TN=382 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 18 | 157 | 1 |
| body_of_stomach_normal | 9 | 1 | 182 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8816    0.9663    0.9220       208
     Malignant_lesions     0.9573    0.8920    0.9235       176
body_of_stomach_normal     0.9891    0.9479    0.9681       192

              accuracy                         0.9375       576
             macro avg     0.9427    0.9354    0.9379       576
          weighted avg     0.9406    0.9375    0.9378       576

```