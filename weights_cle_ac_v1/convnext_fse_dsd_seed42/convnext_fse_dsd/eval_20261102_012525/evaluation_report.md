# convnext_fse_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fse_dsd_seed42/convnext_fse_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9833 |
| Accuracy | 0.9859 |
| Macro-AUROC (OvR) | 0.9977 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9860 |
| Balanced Acc | 0.9847 |
| Macro-Sensitivity | 0.9847 |
| Macro-Specificity | 0.9930 |
| Macro-Precision | 0.9819 |
| AUC-PR (Macro) | 0.9917 |
| Cohen's Kappa | 0.9777 |
| MCC | 0.9777 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 27.78M |
| Pure-Inference FPS | 335.7 (2.98 ms/张) |
| End-to-End FPS | 160.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9914 | 0.9896 | 0.9845 | 0.9829 | 0.9973 | 0.9972 | 195 |
| Malignant_lesions | 0.9747 | 0.9914 | 0.9625 | 0.9686 | 0.9942 | 0.9958 | 0.9780 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=2 TN=230 FN=4
- **Malignant_lesions**: TP=77 FP=3 TN=345 FN=2
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 3 | 1 |
| Malignant_lesions | 2 | 77 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9896    0.9795    0.9845       195
     Malignant_lesions     0.9625    0.9747    0.9686        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9859       427
             macro avg     0.9819    0.9847    0.9833       427
          weighted avg     0.9860    0.9859    0.9860       427

```