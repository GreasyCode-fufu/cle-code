# transnext_tiny 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/transnext_tiny/seed43/transnext_tiny/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9653 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9965 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9654 |
| Balanced Acc | 0.9657 |
| Macro-Sensitivity | 0.9657 |
| Macro-Specificity | 0.9827 |
| Macro-Precision | 0.9652 |
| AUC-PR (Macro) | 0.9932 |
| Cohen's Kappa | 0.9478 |
| MCC | 0.9480 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 5.53G |
| FLOPs | 11.07G |
| Params | 27.63M |
| Pure-Inference FPS | 114.8 (8.71 ms/张) |
| End-to-End FPS | 14.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9783 | 0.9614 | 0.9590 | 0.9756 | 0.9953 | 0.9924 | 208 |
| Malignant_lesions | 0.9716 | 0.9725 | 0.9396 | 0.9553 | 0.9873 | 0.9946 | 0.9879 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9974 | 0.9947 | 0.9815 | 0.9846 | 0.9997 | 0.9994 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=8 TN=360 FN=9
- **Malignant_lesions**: TP=171 FP=11 TN=389 FN=5
- **body_of_stomach_normal**: TP=186 FP=1 TN=383 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 9 | 0 |
| Malignant_lesions | 4 | 171 | 1 |
| body_of_stomach_normal | 4 | 2 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9614    0.9567    0.9590       208
     Malignant_lesions     0.9396    0.9716    0.9553       176
body_of_stomach_normal     0.9947    0.9688    0.9815       192

              accuracy                         0.9653       576
             macro avg     0.9652    0.9657    0.9653       576
          weighted avg     0.9658    0.9653    0.9654       576

```