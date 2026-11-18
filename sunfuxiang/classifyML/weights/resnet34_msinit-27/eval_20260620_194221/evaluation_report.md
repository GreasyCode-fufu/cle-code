# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-27/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9195 |
| Accuracy | 0.9180 |
| Macro-AUROC (OvR) | 0.9822 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9175 |
| Balanced Acc | 0.9240 |
| Macro-Sensitivity | 0.9240 |
| Macro-Specificity | 0.9594 |
| Macro-Precision | 0.9188 |
| AUC-PR (Macro) | 0.9682 |
| Cohen's Kappa | 0.8770 |
| MCC | 0.8789 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.78G |
| FLOPs | 7.57G |
| Params | 21.37M |
| Pure-Inference FPS | 994.8 (1.01 ms/张) |
| End-to-End FPS | 187.6 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8438 | 0.9735 | 0.9515 | 0.8944 | 0.9099 | 0.9729 | 0.9654 | 256 |
| Malignant_lesions | 0.9429 | 0.9306 | 0.8609 | 0.9000 | 0.9728 | 0.9770 | 0.9461 | 210 |
| body_of_stomach_normal | 0.9854 | 0.9742 | 0.9439 | 0.9642 | 0.9934 | 0.9969 | 0.9931 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=216 FP=11 TN=404 FN=40
- **Malignant_lesions**: TP=198 FP=32 TN=429 FN=12
- **body_of_stomach_normal**: TP=202 FP=12 TN=454 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 216 | 30 | 10 |
| Malignant_lesions | 10 | 198 | 2 |
| body_of_stomach_normal | 1 | 2 | 202 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9515    0.8438    0.8944       256
     Malignant_lesions     0.8609    0.9429    0.9000       210
body_of_stomach_normal     0.9439    0.9854    0.9642       205

              accuracy                         0.9180       671
             macro avg     0.9188    0.9240    0.9195       671
          weighted avg     0.9208    0.9180    0.9175       671

```