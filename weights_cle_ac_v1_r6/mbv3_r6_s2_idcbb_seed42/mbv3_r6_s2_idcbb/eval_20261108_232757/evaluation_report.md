# mbv3_r6_s2_idcbb 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_idcbb_seed42/mbv3_r6_s2_idcbb/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9690 |
| Accuracy | 0.9742 |
| Macro-AUROC (OvR) | 0.9982 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9744 |
| Balanced Acc | 0.9762 |
| Macro-Sensitivity | 0.9762 |
| Macro-Specificity | 0.9882 |
| Macro-Precision | 0.9629 |
| AUC-PR (Macro) | 0.9933 |
| Cohen's Kappa | 0.9593 |
| MCC | 0.9597 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 234.73M |
| FLOPs | 469.46M |
| Params | 3.26M |
| Pure-Inference FPS | 2656.3 (0.38 ms/张) |
| End-to-End FPS | 200.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9538 | 0.9914 | 0.9894 | 0.9713 | 0.9623 | 0.9982 | 0.9980 | 195 |
| Malignant_lesions | 0.9747 | 0.9770 | 0.9059 | 0.9390 | 0.9942 | 0.9964 | 0.9818 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=186 FP=2 TN=230 FN=9
- **Malignant_lesions**: TP=77 FP=8 TN=340 FN=2
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 186 | 8 | 1 |
| Malignant_lesions | 2 | 77 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9894    0.9538    0.9713       195
     Malignant_lesions     0.9059    0.9747    0.9390        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9742       427
             macro avg     0.9629    0.9762    0.9690       427
          weighted avg     0.9754    0.9742    0.9744       427

```