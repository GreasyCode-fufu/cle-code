# convnext_egb_scg 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/cle_net_rerun_codex_20261109_v2/results/convnext_egb_scg_seed43/convnext_egb_scg/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9550 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9936 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9549 |
| Balanced Acc | 0.9542 |
| Macro-Sensitivity | 0.9542 |
| Macro-Specificity | 0.9771 |
| Macro-Precision | 0.9560 |
| AUC-PR (Macro) | 0.9880 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9322 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 1007.2 (0.99 ms/张) |
| End-to-End FPS | 221.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9592 | 0.9299 | 0.9431 | 0.9751 | 0.9913 | 0.9840 | 208 |
| Malignant_lesions | 0.9318 | 0.9825 | 0.9591 | 0.9452 | 0.9704 | 0.9909 | 0.9825 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9896 | 0.9791 | 0.9765 | 0.9870 | 0.9986 | 0.9975 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=15 TN=353 FN=9
- **Malignant_lesions**: TP=164 FP=7 TN=393 FN=12
- **body_of_stomach_normal**: TP=187 FP=4 TN=380 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 10 | 164 | 2 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9299    0.9567    0.9431       208
     Malignant_lesions     0.9591    0.9318    0.9452       176
body_of_stomach_normal     0.9791    0.9740    0.9765       192

              accuracy                         0.9549       576
             macro avg     0.9560    0.9542    0.9550       576
          weighted avg     0.9552    0.9549    0.9549       576

```