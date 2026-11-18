# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8918 |
| Accuracy | 0.8897 |
| Macro-AUROC (OvR) | 0.9768 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8894 |
| Balanced Acc | 0.8971 |
| Macro-Sensitivity | 0.8971 |
| Macro-Specificity | 0.9456 |
| Macro-Precision | 0.8938 |
| AUC-PR (Macro) | 0.9600 |
| Cohen's Kappa | 0.8347 |
| MCC | 0.8385 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.71G |
| FLOPs | 7.41G |
| Params | 21.30M |
| Pure-Inference FPS | 3022.2 (0.33 ms/张) |
| End-to-End FPS | 227.5 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7969 | 0.9687 | 0.9401 | 0.8626 | 0.8855 | 0.9671 | 0.9560 | 256 |
| Malignant_lesions | 0.9333 | 0.8959 | 0.8033 | 0.8634 | 0.9672 | 0.9688 | 0.9367 | 210 |
| body_of_stomach_normal | 0.9610 | 0.9721 | 0.9381 | 0.9494 | 0.9826 | 0.9943 | 0.9873 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=13 TN=402 FN=52
- **Malignant_lesions**: TP=196 FP=48 TN=413 FN=14
- **body_of_stomach_normal**: TP=197 FP=13 TN=453 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 43 | 9 |
| Malignant_lesions | 10 | 196 | 4 |
| body_of_stomach_normal | 3 | 5 | 197 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9401    0.7969    0.8626       256
     Malignant_lesions     0.8033    0.9333    0.8634       210
body_of_stomach_normal     0.9381    0.9610    0.9494       205

              accuracy                         0.8897       671
             macro avg     0.8938    0.8971    0.8918       671
          weighted avg     0.8967    0.8897    0.8894       671

```