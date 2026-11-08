# mbv3_r6_s2_edge 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_edge_seed42/mbv3_r6_s2_edge/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9480 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9943 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9480 |
| Balanced Acc | 0.9469 |
| Macro-Sensitivity | 0.9469 |
| Macro-Specificity | 0.9736 |
| Macro-Precision | 0.9496 |
| AUC-PR (Macro) | 0.9895 |
| Cohen's Kappa | 0.9216 |
| MCC | 0.9219 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 3836.2 (0.26 ms/张) |
| End-to-End FPS | 233.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9484 | 0.9124 | 0.9318 | 0.9721 | 0.9912 | 0.9849 | 208 |
| Malignant_lesions | 0.9148 | 0.9775 | 0.9471 | 0.9306 | 0.9631 | 0.9930 | 0.9858 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9948 | 0.9894 | 0.9816 | 0.9871 | 0.9987 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=19 TN=349 FN=10
- **Malignant_lesions**: TP=161 FP=9 TN=391 FN=15
- **body_of_stomach_normal**: TP=187 FP=2 TN=382 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 8 | 2 |
| Malignant_lesions | 15 | 161 | 0 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9124    0.9519    0.9318       208
     Malignant_lesions     0.9471    0.9148    0.9306       176
body_of_stomach_normal     0.9894    0.9740    0.9816       192

              accuracy                         0.9479       576
             macro avg     0.9496    0.9469    0.9480       576
          weighted avg     0.9487    0.9479    0.9480       576

```