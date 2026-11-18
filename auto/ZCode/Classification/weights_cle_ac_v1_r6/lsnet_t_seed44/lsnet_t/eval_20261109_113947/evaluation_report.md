# lsnet_t 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/lsnet_t_seed44/lsnet_t/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9301 |
| Accuracy | 0.9306 |
| Macro-AUROC (OvR) | 0.9895 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9302 |
| Balanced Acc | 0.9297 |
| Macro-Sensitivity | 0.9297 |
| Macro-Specificity | 0.9649 |
| Macro-Precision | 0.9316 |
| AUC-PR (Macro) | 0.9800 |
| Cohen's Kappa | 0.8955 |
| MCC | 0.8960 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 320.32M |
| FLOPs | 640.63M |
| Params | 11.02M |
| Pure-Inference FPS | 5400.9 (0.19 ms/张) |
| End-to-End FPS | 248.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9183 | 0.9538 | 0.9183 | 0.9183 | 0.9538 | 0.9845 | 0.9733 | 208 |
| Malignant_lesions | 0.8864 | 0.9775 | 0.9455 | 0.9150 | 0.9513 | 0.9854 | 0.9694 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9635 | 0.9310 | 0.9570 | 0.9920 | 0.9986 | 0.9973 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=17 TN=351 FN=17
- **Malignant_lesions**: TP=156 FP=9 TN=391 FN=20
- **body_of_stomach_normal**: TP=189 FP=14 TN=370 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 9 | 8 |
| Malignant_lesions | 14 | 156 | 6 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9183    0.9183    0.9183       208
     Malignant_lesions     0.9455    0.8864    0.9150       176
body_of_stomach_normal     0.9310    0.9844    0.9570       192

              accuracy                         0.9306       576
             macro avg     0.9316    0.9297    0.9301       576
          weighted avg     0.9308    0.9306    0.9302       576

```