# convnext_fgc_dsd_fse 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_dsd_fse_seed43/convnext_fgc_dsd_fse/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9771 |
| Accuracy | 0.9813 |
| Macro-AUROC (OvR) | 0.9984 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9812 |
| Balanced Acc | 0.9738 |
| Macro-Sensitivity | 0.9738 |
| Macro-Specificity | 0.9897 |
| Macro-Precision | 0.9806 |
| AUC-PR (Macro) | 0.9964 |
| Cohen's Kappa | 0.9701 |
| MCC | 0.9702 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 278.7 (3.59 ms/张) |
| End-to-End FPS | 146.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9784 | 0.9746 | 0.9796 | 0.9870 | 0.9974 | 0.9974 | 195 |
| Malignant_lesions | 0.9367 | 0.9943 | 0.9737 | 0.9548 | 0.9858 | 0.9980 | 0.9921 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 0.9998 | 0.9996 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=5 TN=227 FN=3
- **Malignant_lesions**: TP=74 FP=2 TN=346 FN=5
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 2 | 1 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9746    0.9846    0.9796       195
     Malignant_lesions     0.9737    0.9367    0.9548        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9813       427
             macro avg     0.9806    0.9738    0.9771       427
          weighted avg     0.9812    0.9813    0.9812       427

```