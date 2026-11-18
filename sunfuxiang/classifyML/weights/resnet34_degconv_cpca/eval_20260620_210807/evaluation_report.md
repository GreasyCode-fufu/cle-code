# resnet34_degconv_cpca 评估报告

- 权重: `./weights/resnet34_degconv_cpca/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9077 |
| Accuracy | 0.9061 |
| Macro-AUROC (OvR) | 0.9760 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9057 |
| Balanced Acc | 0.9110 |
| Macro-Sensitivity | 0.9110 |
| Macro-Specificity | 0.9531 |
| Macro-Precision | 0.9064 |
| AUC-PR (Macro) | 0.9565 |
| Cohen's Kappa | 0.8589 |
| MCC | 0.8599 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.87G |
| FLOPs | 7.74G |
| Params | 21.95M |
| Pure-Inference FPS | 976.8 (1.02 ms/张) |
| End-to-End FPS | 198.4 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8477 | 0.9566 | 0.9234 | 0.8839 | 0.9106 | 0.9636 | 0.9522 | 256 |
| Malignant_lesions | 0.9048 | 0.9284 | 0.8520 | 0.8776 | 0.9554 | 0.9671 | 0.9236 | 210 |
| body_of_stomach_normal | 0.9805 | 0.9742 | 0.9437 | 0.9617 | 0.9913 | 0.9974 | 0.9937 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=217 FP=18 TN=397 FN=39
- **Malignant_lesions**: TP=190 FP=33 TN=428 FN=20
- **body_of_stomach_normal**: TP=201 FP=12 TN=454 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 217 | 31 | 8 |
| Malignant_lesions | 16 | 190 | 4 |
| body_of_stomach_normal | 2 | 2 | 201 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9234    0.8477    0.8839       256
     Malignant_lesions     0.8520    0.9048    0.8776       210
body_of_stomach_normal     0.9437    0.9805    0.9617       205

              accuracy                         0.9061       671
             macro avg     0.9064    0.9110    0.9077       671
          weighted avg     0.9073    0.9061    0.9057       671

```