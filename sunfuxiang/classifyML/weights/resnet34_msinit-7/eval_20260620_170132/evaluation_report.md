# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-7/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8913 |
| Accuracy | 0.8897 |
| Macro-AUROC (OvR) | 0.9710 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8894 |
| Balanced Acc | 0.8922 |
| Macro-Sensitivity | 0.8922 |
| Macro-Specificity | 0.9441 |
| Macro-Precision | 0.8905 |
| AUC-PR (Macro) | 0.9442 |
| Cohen's Kappa | 0.8338 |
| MCC | 0.8339 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.49G |
| FLOPs | 8.97G |
| Params | 22.28M |
| Pure-Inference FPS | 2780.9 (0.36 ms/张) |
| End-to-End FPS | 223.3 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8633 | 0.9253 | 0.8770 | 0.8701 | 0.9165 | 0.9602 | 0.9451 | 256 |
| Malignant_lesions | 0.8524 | 0.9349 | 0.8565 | 0.8544 | 0.9329 | 0.9592 | 0.8986 | 210 |
| body_of_stomach_normal | 0.9610 | 0.9721 | 0.9381 | 0.9494 | 0.9826 | 0.9935 | 0.9888 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=221 FP=31 TN=384 FN=35
- **Malignant_lesions**: TP=179 FP=30 TN=431 FN=31
- **body_of_stomach_normal**: TP=197 FP=13 TN=453 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 221 | 25 | 10 |
| Malignant_lesions | 28 | 179 | 3 |
| body_of_stomach_normal | 3 | 5 | 197 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8770    0.8633    0.8701       256
     Malignant_lesions     0.8565    0.8524    0.8544       210
body_of_stomach_normal     0.9381    0.9610    0.9494       205

              accuracy                         0.8897       671
             macro avg     0.8905    0.8922    0.8913       671
          weighted avg     0.8892    0.8897    0.8894       671

```