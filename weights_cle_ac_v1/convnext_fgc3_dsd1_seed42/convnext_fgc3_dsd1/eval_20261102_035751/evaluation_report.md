# convnext_fgc3_dsd1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc3_dsd1_seed42/convnext_fgc3_dsd1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9290 |
| Accuracy | 0.9288 |
| Macro-AUROC (OvR) | 0.9895 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9292 |
| Balanced Acc | 0.9270 |
| Macro-Sensitivity | 0.9270 |
| Macro-Specificity | 0.9638 |
| Macro-Precision | 0.9325 |
| AUC-PR (Macro) | 0.9818 |
| Cohen's Kappa | 0.8928 |
| MCC | 0.8937 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.39G |
| FLOPs | 8.78G |
| Params | 27.62M |
| Pure-Inference FPS | 1052.4 (0.95 ms/张) |
| End-to-End FPS | 241.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9266 | 0.8800 | 0.9145 | 0.9715 | 0.9866 | 0.9775 | 208 |
| Malignant_lesions | 0.8864 | 0.9700 | 0.9286 | 0.9070 | 0.9510 | 0.9843 | 0.9719 | 176 |
| body_of_stomach_normal | 0.9427 | 0.9948 | 0.9891 | 0.9653 | 0.9720 | 0.9977 | 0.9959 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=27 TN=341 FN=10
- **Malignant_lesions**: TP=156 FP=12 TN=388 FN=20
- **body_of_stomach_normal**: TP=181 FP=2 TN=382 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 9 | 1 |
| Malignant_lesions | 19 | 156 | 1 |
| body_of_stomach_normal | 8 | 3 | 181 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8800    0.9519    0.9145       208
     Malignant_lesions     0.9286    0.8864    0.9070       176
body_of_stomach_normal     0.9891    0.9427    0.9653       192

              accuracy                         0.9288       576
             macro avg     0.9325    0.9270    0.9290       576
          weighted avg     0.9312    0.9288    0.9292       576

```