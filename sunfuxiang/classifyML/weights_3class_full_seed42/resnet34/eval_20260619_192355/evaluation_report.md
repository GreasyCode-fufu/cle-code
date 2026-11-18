# resnet34 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed42/resnet34/resnet34_3class_full_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8781 |
| Accuracy | 0.8763 |
| Macro-AUROC (OvR) | 0.9643 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8761 |
| Balanced Acc | 0.8785 |
| Macro-Sensitivity | 0.8785 |
| Macro-Specificity | 0.9372 |
| Macro-Precision | 0.8778 |
| AUC-PR (Macro) | 0.9318 |
| Cohen's Kappa | 0.8135 |
| MCC | 0.8135 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3786.3 (0.26 ms/张) |
| End-to-End FPS | 237.3 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8555 | 0.9133 | 0.8588 | 0.8571 | 0.9111 | 0.9506 | 0.9380 | 256 |
| Malignant_lesions | 0.8238 | 0.9219 | 0.8278 | 0.8258 | 0.9199 | 0.9480 | 0.8715 | 210 |
| body_of_stomach_normal | 0.9561 | 0.9764 | 0.9469 | 0.9515 | 0.9806 | 0.9944 | 0.9858 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=219 FP=36 TN=379 FN=37
- **Malignant_lesions**: TP=173 FP=36 TN=425 FN=37
- **body_of_stomach_normal**: TP=196 FP=11 TN=455 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 219 | 29 | 8 |
| Malignant_lesions | 34 | 173 | 3 |
| body_of_stomach_normal | 2 | 7 | 196 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8588    0.8555    0.8571       256
     Malignant_lesions     0.8278    0.8238    0.8258       210
body_of_stomach_normal     0.9469    0.9561    0.9515       205

              accuracy                         0.8763       671
             macro avg     0.8778    0.8785    0.8781       671
          weighted avg     0.8760    0.8763    0.8761       671

```