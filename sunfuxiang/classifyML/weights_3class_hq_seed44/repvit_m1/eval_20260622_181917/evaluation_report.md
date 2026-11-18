# repvit_m1 评估报告

- 权重: `weights_3class_hq_seed44/repvit_m1/repvit_m1_3class_hq_seed44_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8765 |
| Accuracy | 0.8763 |
| Macro-AUROC (OvR) | 0.9643 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8754 |
| Balanced Acc | 0.8743 |
| Macro-Sensitivity | 0.8743 |
| Macro-Specificity | 0.9362 |
| Macro-Precision | 0.8837 |
| AUC-PR (Macro) | 0.9382 |
| Cohen's Kappa | 0.8127 |
| MCC | 0.8150 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 3838.4 (0.26 ms/张) |
| End-to-End FPS | 214.3 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9102 | 0.8819 | 0.8262 | 0.8662 | 0.9409 | 0.9493 | 0.9270 | 256 |
| Malignant_lesions | 0.7762 | 0.9653 | 0.9106 | 0.8380 | 0.9045 | 0.9538 | 0.9148 | 210 |
| body_of_stomach_normal | 0.9366 | 0.9614 | 0.9143 | 0.9253 | 0.9718 | 0.9897 | 0.9730 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=233 FP=49 TN=366 FN=23
- **Malignant_lesions**: TP=163 FP=16 TN=445 FN=47
- **body_of_stomach_normal**: TP=192 FP=18 TN=448 FN=13

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 233 | 12 | 11 |
| Malignant_lesions | 40 | 163 | 7 |
| body_of_stomach_normal | 9 | 4 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8262    0.9102    0.8662       256
     Malignant_lesions     0.9106    0.7762    0.8380       210
body_of_stomach_normal     0.9143    0.9366    0.9253       205

              accuracy                         0.8763       671
             macro avg     0.8837    0.8743    0.8765       671
          weighted avg     0.8795    0.8763    0.8754       671

```