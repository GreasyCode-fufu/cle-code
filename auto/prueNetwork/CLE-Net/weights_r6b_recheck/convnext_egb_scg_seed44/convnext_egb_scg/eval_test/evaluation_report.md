# convnext_egb_scg 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/weights_r6b_recheck/convnext_egb_scg_seed44/convnext_egb_scg/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9446 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9938 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9448 |
| Balanced Acc | 0.9452 |
| Macro-Sensitivity | 0.9452 |
| Macro-Specificity | 0.9724 |
| Macro-Precision | 0.9451 |
| AUC-PR (Macro) | 0.9885 |
| Cohen's Kappa | 0.9166 |
| MCC | 0.9171 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 454.6 (2.20 ms/张) |
| End-to-End FPS | 11.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9674 | 0.9417 | 0.9372 | 0.9622 | 0.9911 | 0.9880 | 208 |
| Malignant_lesions | 0.9602 | 0.9525 | 0.8989 | 0.9286 | 0.9820 | 0.9918 | 0.9813 | 176 |
| body_of_stomach_normal | 0.9427 | 0.9974 | 0.9945 | 0.9679 | 0.9721 | 0.9984 | 0.9962 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=12 TN=356 FN=14
- **Malignant_lesions**: TP=169 FP=19 TN=381 FN=7
- **body_of_stomach_normal**: TP=181 FP=1 TN=383 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 13 | 1 |
| Malignant_lesions | 7 | 169 | 0 |
| body_of_stomach_normal | 5 | 6 | 181 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9417    0.9327    0.9372       208
     Malignant_lesions     0.8989    0.9602    0.9286       176
body_of_stomach_normal     0.9945    0.9427    0.9679       192

              accuracy                         0.9444       576
             macro avg     0.9451    0.9452    0.9446       576
          weighted avg     0.9463    0.9444    0.9448       576

```