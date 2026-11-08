# convnext_v5sd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5sd_seed43/convnext_v5sd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9529 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9954 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9530 |
| Balanced Acc | 0.9550 |
| Macro-Sensitivity | 0.9550 |
| Macro-Specificity | 0.9770 |
| Macro-Precision | 0.9527 |
| AUC-PR (Macro) | 0.9908 |
| Cohen's Kappa | 0.9297 |
| MCC | 0.9306 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 366.1 (2.73 ms/张) |
| End-to-End FPS | 170.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8990 | 0.9864 | 0.9740 | 0.9350 | 0.9453 | 0.9936 | 0.9901 | 208 |
| Malignant_lesions | 0.9659 | 0.9525 | 0.8995 | 0.9315 | 0.9845 | 0.9927 | 0.9825 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9922 | 0.9846 | 0.9922 | 1.0000 | 0.9999 | 0.9998 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=5 TN=363 FN=21
- **Malignant_lesions**: TP=170 FP=19 TN=381 FN=6
- **body_of_stomach_normal**: TP=192 FP=3 TN=381 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 19 | 2 |
| Malignant_lesions | 5 | 170 | 1 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9740    0.8990    0.9350       208
     Malignant_lesions     0.8995    0.9659    0.9315       176
body_of_stomach_normal     0.9846    1.0000    0.9922       192

              accuracy                         0.9531       576
             macro avg     0.9527    0.9550    0.9529       576
          weighted avg     0.9548    0.9531    0.9530       576

```