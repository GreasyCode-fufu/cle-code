# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz16bn_seed42/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9327 |
| Accuracy | 0.9323 |
| Macro-AUROC (OvR) | 0.9921 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9326 |
| Balanced Acc | 0.9296 |
| Macro-Sensitivity | 0.9296 |
| Macro-Specificity | 0.9651 |
| Macro-Precision | 0.9395 |
| AUC-PR (Macro) | 0.9857 |
| Cohen's Kappa | 0.8979 |
| MCC | 0.8998 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4494.0 (0.22 ms/张) |
| End-to-End FPS | 85.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9158 | 0.8664 | 0.9136 | 0.9797 | 0.9888 | 0.9809 | 208 |
| Malignant_lesions | 0.8693 | 0.9875 | 0.9684 | 0.9162 | 0.9450 | 0.9907 | 0.9815 | 176 |
| body_of_stomach_normal | 0.9531 | 0.9922 | 0.9839 | 0.9683 | 0.9769 | 0.9967 | 0.9945 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=31 TN=337 FN=7
- **Malignant_lesions**: TP=153 FP=5 TN=395 FN=23
- **body_of_stomach_normal**: TP=183 FP=3 TN=381 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 22 | 153 | 1 |
| body_of_stomach_normal | 9 | 0 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8664    0.9663    0.9136       208
     Malignant_lesions     0.9684    0.8693    0.9162       176
body_of_stomach_normal     0.9839    0.9531    0.9683       192

              accuracy                         0.9323       576
             macro avg     0.9395    0.9296    0.9327       576
          weighted avg     0.9367    0.9323    0.9326       576

```