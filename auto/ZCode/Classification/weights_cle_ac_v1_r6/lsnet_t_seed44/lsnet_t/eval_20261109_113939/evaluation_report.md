# lsnet_t 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/lsnet_t_seed44/lsnet_t/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9679 |
| Accuracy | 0.9742 |
| Macro-AUROC (OvR) | 0.9970 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9741 |
| Balanced Acc | 0.9661 |
| Macro-Sensitivity | 0.9661 |
| Macro-Specificity | 0.9866 |
| Macro-Precision | 0.9698 |
| AUC-PR (Macro) | 0.9900 |
| Cohen's Kappa | 0.9590 |
| MCC | 0.9590 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 320.32M |
| FLOPs | 640.63M |
| Params | 11.02M |
| Pure-Inference FPS | 3257.5 (0.31 ms/张) |
| End-to-End FPS | 208.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9784 | 0.9744 | 0.9744 | 0.9784 | 0.9969 | 0.9963 | 195 |
| Malignant_lesions | 0.9241 | 0.9885 | 0.9481 | 0.9359 | 0.9829 | 0.9942 | 0.9738 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 1.0000 | 0.9999 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=5 TN=227 FN=5
- **Malignant_lesions**: TP=73 FP=4 TN=344 FN=6
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 4 | 1 |
| Malignant_lesions | 5 | 73 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9744    0.9744    0.9744       195
     Malignant_lesions     0.9481    0.9241    0.9359        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9742       427
             macro avg     0.9698    0.9661    0.9679       427
          weighted avg     0.9741    0.9742    0.9741       427

```