# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-11/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8898 |
| Accuracy | 0.8882 |
| Macro-AUROC (OvR) | 0.9725 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8877 |
| Balanced Acc | 0.8938 |
| Macro-Sensitivity | 0.8938 |
| Macro-Specificity | 0.9444 |
| Macro-Precision | 0.8888 |
| AUC-PR (Macro) | 0.9510 |
| Cohen's Kappa | 0.8322 |
| MCC | 0.8337 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.71G |
| FLOPs | 9.42G |
| Params | 24.58M |
| Pure-Inference FPS | 3073.4 (0.33 ms/张) |
| End-to-End FPS | 227.5 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8203 | 0.9542 | 0.9170 | 0.8660 | 0.8959 | 0.9598 | 0.9404 | 256 |
| Malignant_lesions | 0.8905 | 0.9132 | 0.8238 | 0.8558 | 0.9482 | 0.9628 | 0.9249 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9657 | 0.9256 | 0.9476 | 0.9868 | 0.9949 | 0.9877 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=210 FP=19 TN=396 FN=46
- **Malignant_lesions**: TP=187 FP=40 TN=421 FN=23
- **body_of_stomach_normal**: TP=199 FP=16 TN=450 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 210 | 35 | 11 |
| Malignant_lesions | 18 | 187 | 5 |
| body_of_stomach_normal | 1 | 5 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9170    0.8203    0.8660       256
     Malignant_lesions     0.8238    0.8905    0.8558       210
body_of_stomach_normal     0.9256    0.9707    0.9476       205

              accuracy                         0.8882       671
             macro avg     0.8888    0.8938    0.8898       671
          weighted avg     0.8905    0.8882    0.8877       671

```