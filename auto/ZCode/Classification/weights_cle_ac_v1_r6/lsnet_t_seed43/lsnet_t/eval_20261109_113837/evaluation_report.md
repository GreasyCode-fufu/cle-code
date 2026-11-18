# lsnet_t 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/lsnet_t_seed43/lsnet_t/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9545 |
| Accuracy | 0.9625 |
| Macro-AUROC (OvR) | 0.9938 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9623 |
| Balanced Acc | 0.9516 |
| Macro-Sensitivity | 0.9516 |
| Macro-Specificity | 0.9803 |
| Macro-Precision | 0.9576 |
| AUC-PR (Macro) | 0.9849 |
| Cohen's Kappa | 0.9403 |
| MCC | 0.9403 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 320.32M |
| FLOPs | 640.63M |
| Params | 11.02M |
| Pure-Inference FPS | 1930.8 (0.52 ms/张) |
| End-to-End FPS | 225.3 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9698 | 0.9643 | 0.9668 | 0.9740 | 0.9943 | 0.9935 | 195 |
| Malignant_lesions | 0.8987 | 0.9856 | 0.9342 | 0.9161 | 0.9772 | 0.9885 | 0.9637 | 79 |
| body_of_stomach_normal | 0.9869 | 0.9854 | 0.9742 | 0.9805 | 0.9926 | 0.9986 | 0.9974 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=7 TN=225 FN=6
- **Malignant_lesions**: TP=71 FP=5 TN=343 FN=8
- **body_of_stomach_normal**: TP=151 FP=4 TN=270 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 5 | 1 |
| Malignant_lesions | 5 | 71 | 3 |
| body_of_stomach_normal | 2 | 0 | 151 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9643    0.9692    0.9668       195
     Malignant_lesions     0.9342    0.8987    0.9161        79
body_of_stomach_normal     0.9742    0.9869    0.9805       153

              accuracy                         0.9625       427
             macro avg     0.9576    0.9516    0.9545       427
          weighted avg     0.9623    0.9625    0.9623       427

```