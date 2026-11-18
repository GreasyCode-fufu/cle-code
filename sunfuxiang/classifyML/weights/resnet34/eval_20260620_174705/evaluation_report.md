# resnet34 评估报告

- 权重: `./weights/resnet34/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8937 |
| Accuracy | 0.8927 |
| Macro-AUROC (OvR) | 0.9764 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8918 |
| Balanced Acc | 0.8995 |
| Macro-Sensitivity | 0.8995 |
| Macro-Specificity | 0.9471 |
| Macro-Precision | 0.8934 |
| AUC-PR (Macro) | 0.9490 |
| Cohen's Kappa | 0.8391 |
| MCC | 0.8419 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3773.0 (0.27 ms/张) |
| End-to-End FPS | 237.9 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8086 | 0.9687 | 0.9409 | 0.8697 | 0.8914 | 0.9674 | 0.9653 | 256 |
| Malignant_lesions | 0.9143 | 0.9197 | 0.8384 | 0.8747 | 0.9593 | 0.9675 | 0.9097 | 210 |
| body_of_stomach_normal | 0.9756 | 0.9528 | 0.9009 | 0.9368 | 0.9889 | 0.9942 | 0.9719 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=207 FP=13 TN=402 FN=49
- **Malignant_lesions**: TP=192 FP=37 TN=424 FN=18
- **body_of_stomach_normal**: TP=200 FP=22 TN=444 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 207 | 34 | 15 |
| Malignant_lesions | 11 | 192 | 7 |
| body_of_stomach_normal | 2 | 3 | 200 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9409    0.8086    0.8697       256
     Malignant_lesions     0.8384    0.9143    0.8747       210
body_of_stomach_normal     0.9009    0.9756    0.9368       205

              accuracy                         0.8927       671
             macro avg     0.8934    0.8995    0.8937       671
          weighted avg     0.8966    0.8927    0.8918       671

```