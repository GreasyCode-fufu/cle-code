# fast_t0 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed42/fast_t0/fast_t0_3class_full_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8864 |
| Accuracy | 0.8852 |
| Macro-AUROC (OvR) | 0.9693 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8846 |
| Balanced Acc | 0.8866 |
| Macro-Sensitivity | 0.8866 |
| Macro-Specificity | 0.9415 |
| Macro-Precision | 0.8870 |
| AUC-PR (Macro) | 0.9430 |
| Cohen's Kappa | 0.8268 |
| MCC | 0.8271 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 8593.3 (0.12 ms/张) |
| End-to-End FPS | 191.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8750 | 0.9108 | 0.8582 | 0.8665 | 0.9220 | 0.9539 | 0.9377 | 256 |
| Malignant_lesions | 0.8190 | 0.9436 | 0.8687 | 0.8431 | 0.9197 | 0.9587 | 0.9021 | 210 |
| body_of_stomach_normal | 0.9659 | 0.9700 | 0.9340 | 0.9496 | 0.9847 | 0.9953 | 0.9893 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=224 FP=37 TN=378 FN=32
- **Malignant_lesions**: TP=172 FP=26 TN=435 FN=38
- **body_of_stomach_normal**: TP=198 FP=14 TN=452 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 224 | 23 | 9 |
| Malignant_lesions | 33 | 172 | 5 |
| body_of_stomach_normal | 4 | 3 | 198 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8582    0.8750    0.8665       256
     Malignant_lesions     0.8687    0.8190    0.8431       210
body_of_stomach_normal     0.9340    0.9659    0.9496       205

              accuracy                         0.8852       671
             macro avg     0.8870    0.8866    0.8864       671
          weighted avg     0.8846    0.8852    0.8846       671

```