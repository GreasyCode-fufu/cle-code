# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-2/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8650 |
| Accuracy | 0.8629 |
| Macro-AUROC (OvR) | 0.9783 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8607 |
| Balanced Acc | 0.8748 |
| Macro-Sensitivity | 0.8748 |
| Macro-Specificity | 0.9327 |
| Macro-Precision | 0.8733 |
| AUC-PR (Macro) | 0.9628 |
| Cohen's Kappa | 0.7953 |
| MCC | 0.8043 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.80G |
| FLOPs | 7.59G |
| Params | 22.16M |
| Pure-Inference FPS | 3156.1 (0.32 ms/张) |
| End-to-End FPS | 226.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7109 | 0.9735 | 0.9430 | 0.8107 | 0.8452 | 0.9705 | 0.9600 | 256 |
| Malignant_lesions | 0.9429 | 0.8612 | 0.7557 | 0.8390 | 0.9707 | 0.9697 | 0.9406 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9635 | 0.9213 | 0.9454 | 0.9868 | 0.9946 | 0.9879 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=182 FP=11 TN=404 FN=74
- **Malignant_lesions**: TP=198 FP=64 TN=397 FN=12
- **body_of_stomach_normal**: TP=199 FP=17 TN=449 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 182 | 60 | 14 |
| Malignant_lesions | 9 | 198 | 3 |
| body_of_stomach_normal | 2 | 4 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9430    0.7109    0.8107       256
     Malignant_lesions     0.7557    0.9429    0.8390       210
body_of_stomach_normal     0.9213    0.9707    0.9454       205

              accuracy                         0.8629       671
             macro avg     0.8733    0.8748    0.8650       671
          weighted avg     0.8778    0.8629    0.8607       671

```