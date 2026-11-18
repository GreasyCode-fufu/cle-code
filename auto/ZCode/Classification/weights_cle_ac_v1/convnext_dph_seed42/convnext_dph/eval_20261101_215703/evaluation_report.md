# convnext_dph 评估报告

- 权重: `weights_cle_ac_v1/convnext_dph_seed42/convnext_dph/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9651 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9652 |
| Balanced Acc | 0.9658 |
| Macro-Sensitivity | 0.9658 |
| Macro-Specificity | 0.9828 |
| Macro-Precision | 0.9646 |
| AUC-PR (Macro) | 0.9934 |
| Cohen's Kappa | 0.9478 |
| MCC | 0.9480 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 1707.1 (0.59 ms/张) |
| End-to-End FPS | 215.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9837 | 0.9703 | 0.9561 | 0.9679 | 0.9954 | 0.9929 | 208 |
| Malignant_lesions | 0.9602 | 0.9725 | 0.9389 | 0.9494 | 0.9823 | 0.9957 | 0.9883 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9922 | 0.9845 | 0.9896 | 0.9974 | 0.9994 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=6 TN=362 FN=12
- **Malignant_lesions**: TP=169 FP=11 TN=389 FN=7
- **body_of_stomach_normal**: TP=191 FP=3 TN=381 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 11 | 1 |
| Malignant_lesions | 5 | 169 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9703    0.9423    0.9561       208
     Malignant_lesions     0.9389    0.9602    0.9494       176
body_of_stomach_normal     0.9845    0.9948    0.9896       192

              accuracy                         0.9653       576
             macro avg     0.9646    0.9658    0.9651       576
          weighted avg     0.9654    0.9653    0.9652       576

```