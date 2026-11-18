# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/ghostnet_seed45/ghostnet/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9633 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9970 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9695 |
| Balanced Acc | 0.9652 |
| Macro-Sensitivity | 0.9652 |
| Macro-Specificity | 0.9849 |
| Macro-Precision | 0.9616 |
| AUC-PR (Macro) | 0.9886 |
| Cohen's Kappa | 0.9517 |
| MCC | 0.9518 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 2125.7 (0.47 ms/张) |
| End-to-End FPS | 202.1 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9828 | 0.9791 | 0.9689 | 0.9661 | 0.9966 | 0.9964 | 195 |
| Malignant_lesions | 0.9367 | 0.9828 | 0.9250 | 0.9308 | 0.9856 | 0.9945 | 0.9694 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9891 | 0.9808 | 0.9903 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=4 TN=228 FN=8
- **Malignant_lesions**: TP=74 FP=6 TN=342 FN=5
- **body_of_stomach_normal**: TP=153 FP=3 TN=271 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 6 | 2 |
| Malignant_lesions | 4 | 74 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9791    0.9590    0.9689       195
     Malignant_lesions     0.9250    0.9367    0.9308        79
body_of_stomach_normal     0.9808    1.0000    0.9903       153

              accuracy                         0.9696       427
             macro avg     0.9616    0.9652    0.9633       427
          weighted avg     0.9697    0.9696    0.9695       427

```