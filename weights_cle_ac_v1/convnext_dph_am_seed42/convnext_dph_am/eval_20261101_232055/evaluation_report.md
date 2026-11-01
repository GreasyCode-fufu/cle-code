# convnext_dph_am 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_am_seed42/convnext_dph_am/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9788 |
| Accuracy | 0.9836 |
| Macro-AUROC (OvR) | 0.9992 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9835 |
| Balanced Acc | 0.9750 |
| Macro-Sensitivity | 0.9750 |
| Macro-Specificity | 0.9909 |
| Macro-Precision | 0.9829 |
| AUC-PR (Macro) | 0.9975 |
| Cohen's Kappa | 0.9738 |
| MCC | 0.9740 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 29.87M |
| Pure-Inference FPS | 458.3 (2.18 ms/张) |
| End-to-End FPS | 183.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9949 | 0.9784 | 0.9749 | 0.9848 | 0.9956 | 0.9992 | 0.9990 | 195 |
| Malignant_lesions | 0.9367 | 0.9943 | 0.9737 | 0.9548 | 0.9858 | 0.9984 | 0.9934 | 79 |
| body_of_stomach_normal | 0.9935 | 1.0000 | 1.0000 | 0.9967 | 0.9964 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=5 TN=227 FN=1
- **Malignant_lesions**: TP=74 FP=2 TN=346 FN=5
- **body_of_stomach_normal**: TP=152 FP=0 TN=274 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 1 | 0 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 1 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9749    0.9949    0.9848       195
     Malignant_lesions     0.9737    0.9367    0.9548        79
body_of_stomach_normal     1.0000    0.9935    0.9967       153

              accuracy                         0.9836       427
             macro avg     0.9829    0.9750    0.9788       427
          weighted avg     0.9837    0.9836    0.9835       427

```