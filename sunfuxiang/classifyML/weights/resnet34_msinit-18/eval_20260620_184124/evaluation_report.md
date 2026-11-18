# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-18/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9088 |
| Accuracy | 0.9076 |
| Macro-AUROC (OvR) | 0.9782 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9074 |
| Balanced Acc | 0.9118 |
| Macro-Sensitivity | 0.9118 |
| Macro-Specificity | 0.9540 |
| Macro-Precision | 0.9076 |
| AUC-PR (Macro) | 0.9621 |
| Cohen's Kappa | 0.8611 |
| MCC | 0.8621 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.31G |
| FLOPs | 8.63G |
| Params | 21.86M |
| Pure-Inference FPS | 2693.3 (0.37 ms/张) |
| End-to-End FPS | 240.5 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8555 | 0.9614 | 0.9319 | 0.8921 | 0.9151 | 0.9696 | 0.9608 | 256 |
| Malignant_lesions | 0.9190 | 0.9328 | 0.8616 | 0.8894 | 0.9620 | 0.9724 | 0.9489 | 210 |
| body_of_stomach_normal | 0.9610 | 0.9678 | 0.9292 | 0.9448 | 0.9826 | 0.9927 | 0.9766 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=219 FP=16 TN=399 FN=37
- **Malignant_lesions**: TP=193 FP=31 TN=430 FN=17
- **body_of_stomach_normal**: TP=197 FP=15 TN=451 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 219 | 25 | 12 |
| Malignant_lesions | 14 | 193 | 3 |
| body_of_stomach_normal | 2 | 6 | 197 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9319    0.8555    0.8921       256
     Malignant_lesions     0.8616    0.9190    0.8894       210
body_of_stomach_normal     0.9292    0.9610    0.9448       205

              accuracy                         0.9076       671
             macro avg     0.9076    0.9118    0.9088       671
          weighted avg     0.9091    0.9076    0.9074       671

```