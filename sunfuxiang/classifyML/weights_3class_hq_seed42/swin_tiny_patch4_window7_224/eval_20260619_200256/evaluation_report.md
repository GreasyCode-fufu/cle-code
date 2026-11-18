# swin_tiny_patch4_window7_224 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_hq_seed42/swin_tiny_patch4_window7_224/swin_tiny_patch4_window7_224_3class_hq_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8959 |
| Accuracy | 0.8942 |
| Macro-AUROC (OvR) | 0.9753 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8937 |
| Balanced Acc | 0.9007 |
| Macro-Sensitivity | 0.9007 |
| Macro-Specificity | 0.9476 |
| Macro-Precision | 0.8961 |
| AUC-PR (Macro) | 0.9545 |
| Cohen's Kappa | 0.8413 |
| MCC | 0.8438 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.92G |
| Params | 27.94M |
| Pure-Inference FPS | 1528.5 (0.65 ms/张) |
| End-to-End FPS | 252.2 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8125 | 0.9639 | 0.9327 | 0.8685 | 0.8929 | 0.9680 | 0.9547 | 256 |
| Malignant_lesions | 0.9286 | 0.9111 | 0.8263 | 0.8744 | 0.9655 | 0.9657 | 0.9261 | 210 |
| body_of_stomach_normal | 0.9610 | 0.9678 | 0.9292 | 0.9448 | 0.9826 | 0.9923 | 0.9826 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=208 FP=15 TN=400 FN=48
- **Malignant_lesions**: TP=195 FP=41 TN=420 FN=15
- **body_of_stomach_normal**: TP=197 FP=15 TN=451 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 208 | 38 | 10 |
| Malignant_lesions | 10 | 195 | 5 |
| body_of_stomach_normal | 5 | 3 | 197 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9327    0.8125    0.8685       256
     Malignant_lesions     0.8263    0.9286    0.8744       210
body_of_stomach_normal     0.9292    0.9610    0.9448       205

              accuracy                         0.8942       671
             macro avg     0.8961    0.9007    0.8959       671
          weighted avg     0.8983    0.8942    0.8937       671

```