# fast_t0 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_hq_seed42/fast_t0/fast_t0_3class_hq_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8984 |
| Accuracy | 0.8972 |
| Macro-AUROC (OvR) | 0.9767 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8968 |
| Balanced Acc | 0.9005 |
| Macro-Sensitivity | 0.9005 |
| Macro-Specificity | 0.9483 |
| Macro-Precision | 0.8970 |
| AUC-PR (Macro) | 0.9591 |
| Cohen's Kappa | 0.8452 |
| MCC | 0.8456 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 9146.6 (0.11 ms/张) |
| End-to-End FPS | 230.2 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8594 | 0.9422 | 0.9016 | 0.8800 | 0.9157 | 0.9644 | 0.9523 | 256 |
| Malignant_lesions | 0.8762 | 0.9349 | 0.8598 | 0.8679 | 0.9431 | 0.9700 | 0.9335 | 210 |
| body_of_stomach_normal | 0.9659 | 0.9678 | 0.9296 | 0.9474 | 0.9847 | 0.9958 | 0.9914 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=220 FP=24 TN=391 FN=36
- **Malignant_lesions**: TP=184 FP=30 TN=431 FN=26
- **body_of_stomach_normal**: TP=198 FP=15 TN=451 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 220 | 26 | 10 |
| Malignant_lesions | 21 | 184 | 5 |
| body_of_stomach_normal | 3 | 4 | 198 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9016    0.8594    0.8800       256
     Malignant_lesions     0.8598    0.8762    0.8679       210
body_of_stomach_normal     0.9296    0.9659    0.9474       205

              accuracy                         0.8972       671
             macro avg     0.8970    0.9005    0.8984       671
          weighted avg     0.8971    0.8972    0.8968       671

```