# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-1/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8946 |
| Accuracy | 0.8927 |
| Macro-AUROC (OvR) | 0.9798 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8926 |
| Balanced Acc | 0.9002 |
| Macro-Sensitivity | 0.9002 |
| Macro-Specificity | 0.9473 |
| Macro-Precision | 0.8982 |
| AUC-PR (Macro) | 0.9637 |
| Cohen's Kappa | 0.8393 |
| MCC | 0.8440 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.86G |
| Params | 21.53M |
| Pure-Inference FPS | 3092.4 (0.32 ms/张) |
| End-to-End FPS | 202.2 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7969 | 0.9783 | 0.9577 | 0.8699 | 0.8865 | 0.9720 | 0.9646 | 256 |
| Malignant_lesions | 0.9476 | 0.8915 | 0.7992 | 0.8671 | 0.9739 | 0.9735 | 0.9406 | 210 |
| body_of_stomach_normal | 0.9561 | 0.9721 | 0.9378 | 0.9469 | 0.9805 | 0.9941 | 0.9859 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=9 TN=406 FN=52
- **Malignant_lesions**: TP=199 FP=50 TN=411 FN=11
- **body_of_stomach_normal**: TP=196 FP=13 TN=453 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 42 | 10 |
| Malignant_lesions | 8 | 199 | 3 |
| body_of_stomach_normal | 1 | 8 | 196 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9577    0.7969    0.8699       256
     Malignant_lesions     0.7992    0.9476    0.8671       210
body_of_stomach_normal     0.9378    0.9561    0.9469       205

              accuracy                         0.8927       671
             macro avg     0.8982    0.9002    0.8946       671
          weighted avg     0.9020    0.8927    0.8926       671

```