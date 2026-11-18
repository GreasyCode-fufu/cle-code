# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_seed44/fast_t0/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9306 |
| Accuracy | 0.9306 |
| Macro-AUROC (OvR) | 0.9882 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9308 |
| Balanced Acc | 0.9284 |
| Macro-Sensitivity | 0.9284 |
| Macro-Specificity | 0.9646 |
| Macro-Precision | 0.9346 |
| AUC-PR (Macro) | 0.9785 |
| Cohen's Kappa | 0.8954 |
| MCC | 0.8964 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 8581.8 (0.12 ms/张) |
| End-to-End FPS | 101.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9266 | 0.8805 | 0.9171 | 0.9743 | 0.9845 | 0.9744 | 208 |
| Malignant_lesions | 0.8807 | 0.9750 | 0.9394 | 0.9091 | 0.9489 | 0.9815 | 0.9640 | 176 |
| body_of_stomach_normal | 0.9479 | 0.9922 | 0.9838 | 0.9655 | 0.9744 | 0.9985 | 0.9972 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=27 TN=341 FN=9
- **Malignant_lesions**: TP=155 FP=10 TN=390 FN=21
- **body_of_stomach_normal**: TP=182 FP=3 TN=381 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 19 | 155 | 2 |
| body_of_stomach_normal | 8 | 2 | 182 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8805    0.9567    0.9171       208
     Malignant_lesions     0.9394    0.8807    0.9091       176
body_of_stomach_normal     0.9838    0.9479    0.9655       192

              accuracy                         0.9306       576
             macro avg     0.9346    0.9284    0.9306       576
          weighted avg     0.9329    0.9306    0.9308       576

```