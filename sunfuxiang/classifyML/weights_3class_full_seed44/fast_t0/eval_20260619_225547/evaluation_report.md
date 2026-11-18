# fast_t0 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed44/fast_t0/fast_t0_3class_full_seed44_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8677 |
| Accuracy | 0.8674 |
| Macro-AUROC (OvR) | 0.9629 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8655 |
| Balanced Acc | 0.8668 |
| Macro-Sensitivity | 0.8668 |
| Macro-Specificity | 0.9314 |
| Macro-Precision | 0.8746 |
| AUC-PR (Macro) | 0.9364 |
| Cohen's Kappa | 0.7993 |
| MCC | 0.8019 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 9040.4 (0.11 ms/张) |
| End-to-End FPS | 216.7 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8867 | 0.8699 | 0.8078 | 0.8454 | 0.9256 | 0.9424 | 0.9149 | 256 |
| Malignant_lesions | 0.7381 | 0.9566 | 0.8857 | 0.8052 | 0.8891 | 0.9514 | 0.9044 | 210 |
| body_of_stomach_normal | 0.9756 | 0.9678 | 0.9302 | 0.9524 | 0.9890 | 0.9949 | 0.9899 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=227 FP=54 TN=361 FN=29
- **Malignant_lesions**: TP=155 FP=20 TN=441 FN=55
- **body_of_stomach_normal**: TP=200 FP=15 TN=451 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 227 | 18 | 11 |
| Malignant_lesions | 51 | 155 | 4 |
| body_of_stomach_normal | 3 | 2 | 200 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8078    0.8867    0.8454       256
     Malignant_lesions     0.8857    0.7381    0.8052       210
body_of_stomach_normal     0.9302    0.9756    0.9524       205

              accuracy                         0.8674       671
             macro avg     0.8746    0.8668    0.8677       671
          weighted avg     0.8696    0.8674    0.8655       671

```