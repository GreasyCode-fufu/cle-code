# fast_t0 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_hq_seed44/fast_t0/fast_t0_3class_hq_seed44_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8982 |
| Accuracy | 0.8972 |
| Macro-AUROC (OvR) | 0.9739 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8965 |
| Balanced Acc | 0.8979 |
| Macro-Sensitivity | 0.8979 |
| Macro-Specificity | 0.9474 |
| Macro-Precision | 0.8995 |
| AUC-PR (Macro) | 0.9528 |
| Cohen's Kappa | 0.8448 |
| MCC | 0.8452 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 9095.5 (0.11 ms/张) |
| End-to-End FPS | 221.1 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8945 | 0.9157 | 0.8674 | 0.8808 | 0.9337 | 0.9604 | 0.9435 | 256 |
| Malignant_lesions | 0.8286 | 0.9544 | 0.8923 | 0.8593 | 0.9244 | 0.9642 | 0.9214 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9721 | 0.9387 | 0.9544 | 0.9869 | 0.9971 | 0.9936 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=229 FP=35 TN=380 FN=27
- **Malignant_lesions**: TP=174 FP=21 TN=440 FN=36
- **body_of_stomach_normal**: TP=199 FP=13 TN=453 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 229 | 17 | 10 |
| Malignant_lesions | 33 | 174 | 3 |
| body_of_stomach_normal | 2 | 4 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8674    0.8945    0.8808       256
     Malignant_lesions     0.8923    0.8286    0.8593       210
body_of_stomach_normal     0.9387    0.9707    0.9544       205

              accuracy                         0.8972       671
             macro avg     0.8995    0.8979    0.8982       671
          weighted avg     0.8970    0.8972    0.8965       671

```