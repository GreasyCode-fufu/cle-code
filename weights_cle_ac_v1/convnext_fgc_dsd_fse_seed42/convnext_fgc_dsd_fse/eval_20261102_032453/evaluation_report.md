# convnext_fgc_dsd_fse 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_dsd_fse_seed42/convnext_fgc_dsd_fse/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9461 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9464 |
| Balanced Acc | 0.9480 |
| Macro-Sensitivity | 0.9480 |
| Macro-Specificity | 0.9736 |
| Macro-Precision | 0.9462 |
| AUC-PR (Macro) | 0.9902 |
| Cohen's Kappa | 0.9193 |
| MCC | 0.9203 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 307.9 (3.25 ms/张) |
| End-to-End FPS | 155.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8942 | 0.9810 | 0.9637 | 0.9277 | 0.9426 | 0.9934 | 0.9882 | 208 |
| Malignant_lesions | 0.9602 | 0.9425 | 0.8802 | 0.9185 | 0.9818 | 0.9915 | 0.9826 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9974 | 0.9948 | 0.9922 | 0.9948 | 0.9999 | 0.9997 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=186 FP=7 TN=361 FN=22
- **Malignant_lesions**: TP=169 FP=23 TN=377 FN=7
- **body_of_stomach_normal**: TP=190 FP=1 TN=383 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 186 | 21 | 1 |
| Malignant_lesions | 7 | 169 | 0 |
| body_of_stomach_normal | 0 | 2 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9637    0.8942    0.9277       208
     Malignant_lesions     0.8802    0.9602    0.9185       176
body_of_stomach_normal     0.9948    0.9896    0.9922       192

              accuracy                         0.9462       576
             macro avg     0.9462    0.9480    0.9461       576
          weighted avg     0.9486    0.9462    0.9464       576

```