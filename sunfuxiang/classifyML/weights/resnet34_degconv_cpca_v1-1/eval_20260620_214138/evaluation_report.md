# resnet34_degconv_cpca_v1 评估报告

- 权重: `./weights/resnet34_degconv_cpca_v1-1/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8869 |
| Accuracy | 0.8852 |
| Macro-AUROC (OvR) | 0.9779 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8849 |
| Balanced Acc | 0.8892 |
| Macro-Sensitivity | 0.8892 |
| Macro-Specificity | 0.9423 |
| Macro-Precision | 0.8855 |
| AUC-PR (Macro) | 0.9590 |
| Cohen's Kappa | 0.8274 |
| MCC | 0.8278 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.86G |
| FLOPs | 7.73G |
| Params | 22.56M |
| Pure-Inference FPS | 980.5 (1.02 ms/张) |
| End-to-End FPS | 211.1 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8398 | 0.9349 | 0.8884 | 0.8635 | 0.9044 | 0.9697 | 0.9589 | 256 |
| Malignant_lesions | 0.8667 | 0.9241 | 0.8387 | 0.8525 | 0.9383 | 0.9680 | 0.9261 | 210 |
| body_of_stomach_normal | 0.9610 | 0.9678 | 0.9292 | 0.9448 | 0.9826 | 0.9961 | 0.9920 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=215 FP=27 TN=388 FN=41
- **Malignant_lesions**: TP=182 FP=35 TN=426 FN=28
- **body_of_stomach_normal**: TP=197 FP=15 TN=451 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 215 | 30 | 11 |
| Malignant_lesions | 24 | 182 | 4 |
| body_of_stomach_normal | 3 | 5 | 197 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8884    0.8398    0.8635       256
     Malignant_lesions     0.8387    0.8667    0.8525       210
body_of_stomach_normal     0.9292    0.9610    0.9448       205

              accuracy                         0.8852       671
             macro avg     0.8855    0.8892    0.8869       671
          weighted avg     0.8853    0.8852    0.8849       671

```