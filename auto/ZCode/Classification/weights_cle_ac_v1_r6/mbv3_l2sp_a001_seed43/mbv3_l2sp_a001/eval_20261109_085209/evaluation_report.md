# mbv3_l2sp_a001 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_l2sp_a001_seed43/mbv3_l2sp_a001/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9034 |
| Accuracy | 0.9028 |
| Macro-AUROC (OvR) | 0.9890 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9036 |
| Balanced Acc | 0.8995 |
| Macro-Sensitivity | 0.8995 |
| Macro-Specificity | 0.9502 |
| Macro-Precision | 0.9145 |
| AUC-PR (Macro) | 0.9812 |
| Cohen's Kappa | 0.8533 |
| MCC | 0.8572 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3487.2 (0.29 ms/张) |
| End-to-End FPS | 28.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.8832 | 0.8238 | 0.8894 | 0.9789 | 0.9858 | 0.9806 | 208 |
| Malignant_lesions | 0.8466 | 0.9675 | 0.9198 | 0.8817 | 0.9348 | 0.9832 | 0.9663 | 176 |
| body_of_stomach_normal | 0.8854 | 1.0000 | 1.0000 | 0.9392 | 0.9458 | 0.9982 | 0.9967 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=43 TN=325 FN=7
- **Malignant_lesions**: TP=149 FP=13 TN=387 FN=27
- **body_of_stomach_normal**: TP=170 FP=0 TN=384 FN=22

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 7 | 0 |
| Malignant_lesions | 27 | 149 | 0 |
| body_of_stomach_normal | 16 | 6 | 170 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8238    0.9663    0.8894       208
     Malignant_lesions     0.9198    0.8466    0.8817       176
body_of_stomach_normal     1.0000    0.8854    0.9392       192

              accuracy                         0.9028       576
             macro avg     0.9145    0.8995    0.9034       576
          weighted avg     0.9118    0.9028    0.9036       576

```