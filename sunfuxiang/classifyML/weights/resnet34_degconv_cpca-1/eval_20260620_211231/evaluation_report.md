# resnet34_degconv_cpca 评估报告

- 权重: `./weights/resnet34_degconv_cpca-1/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9216 |
| Accuracy | 0.9210 |
| Macro-AUROC (OvR) | 0.9834 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9208 |
| Balanced Acc | 0.9228 |
| Macro-Sensitivity | 0.9228 |
| Macro-Specificity | 0.9603 |
| Macro-Precision | 0.9207 |
| AUC-PR (Macro) | 0.9706 |
| Cohen's Kappa | 0.8810 |
| MCC | 0.8812 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.87G |
| FLOPs | 7.74G |
| Params | 21.95M |
| Pure-Inference FPS | 976.8 (1.02 ms/张) |
| End-to-End FPS | 202.1 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9023 | 0.9542 | 0.9240 | 0.9130 | 0.9406 | 0.9770 | 0.9682 | 256 |
| Malignant_lesions | 0.8952 | 0.9566 | 0.9038 | 0.8995 | 0.9525 | 0.9770 | 0.9536 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9700 | 0.9343 | 0.9522 | 0.9869 | 0.9961 | 0.9899 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=231 FP=19 TN=396 FN=25
- **Malignant_lesions**: TP=188 FP=20 TN=441 FN=22
- **body_of_stomach_normal**: TP=199 FP=14 TN=452 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 231 | 16 | 9 |
| Malignant_lesions | 17 | 188 | 5 |
| body_of_stomach_normal | 2 | 4 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9240    0.9023    0.9130       256
     Malignant_lesions     0.9038    0.8952    0.8995       210
body_of_stomach_normal     0.9343    0.9707    0.9522       205

              accuracy                         0.9210       671
             macro avg     0.9207    0.9228    0.9216       671
          weighted avg     0.9208    0.9210    0.9208       671

```