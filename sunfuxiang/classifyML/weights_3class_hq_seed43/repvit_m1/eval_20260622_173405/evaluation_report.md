# repvit_m1 评估报告

- 权重: `weights_3class_hq_seed43/repvit_m1/repvit_m1_3class_hq_seed43_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8206 |
| Accuracy | 0.8212 |
| Macro-AUROC (OvR) | 0.9447 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8190 |
| Balanced Acc | 0.8242 |
| Macro-Sensitivity | 0.8242 |
| Macro-Specificity | 0.9098 |
| Macro-Precision | 0.8225 |
| AUC-PR (Macro) | 0.9017 |
| Cohen's Kappa | 0.7307 |
| MCC | 0.7328 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 3867.9 (0.26 ms/张) |
| End-to-End FPS | 208.7 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7930 | 0.8867 | 0.8120 | 0.8024 | 0.8741 | 0.9183 | 0.8825 | 256 |
| Malignant_lesions | 0.7333 | 0.9349 | 0.8370 | 0.7817 | 0.8850 | 0.9341 | 0.8605 | 210 |
| body_of_stomach_normal | 0.9463 | 0.9077 | 0.8186 | 0.8778 | 0.9747 | 0.9818 | 0.9620 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=47 TN=368 FN=53
- **Malignant_lesions**: TP=154 FP=30 TN=431 FN=56
- **body_of_stomach_normal**: TP=194 FP=43 TN=423 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 27 | 26 |
| Malignant_lesions | 39 | 154 | 17 |
| body_of_stomach_normal | 8 | 3 | 194 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8120    0.7930    0.8024       256
     Malignant_lesions     0.8370    0.7333    0.7817       210
body_of_stomach_normal     0.8186    0.9463    0.8778       205

              accuracy                         0.8212       671
             macro avg     0.8225    0.8242    0.8206       671
          weighted avg     0.8218    0.8212    0.8190       671

```