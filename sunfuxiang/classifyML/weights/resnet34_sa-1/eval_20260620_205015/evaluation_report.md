# resnet34_sa 评估报告

- 权重: `./weights/resnet34_sa-1/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9132 |
| Accuracy | 0.9121 |
| Macro-AUROC (OvR) | 0.9827 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9114 |
| Balanced Acc | 0.9168 |
| Macro-Sensitivity | 0.9168 |
| Macro-Specificity | 0.9562 |
| Macro-Precision | 0.9115 |
| AUC-PR (Macro) | 0.9680 |
| Cohen's Kappa | 0.8679 |
| MCC | 0.8689 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3740.1 (0.27 ms/张) |
| End-to-End FPS | 231.2 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8555 | 0.9614 | 0.9319 | 0.8921 | 0.9151 | 0.9758 | 0.9665 | 256 |
| Malignant_lesions | 0.9048 | 0.9414 | 0.8756 | 0.8899 | 0.9559 | 0.9748 | 0.9429 | 210 |
| body_of_stomach_normal | 0.9902 | 0.9657 | 0.9269 | 0.9575 | 0.9956 | 0.9975 | 0.9946 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=219 FP=16 TN=399 FN=37
- **Malignant_lesions**: TP=190 FP=27 TN=434 FN=20
- **body_of_stomach_normal**: TP=203 FP=16 TN=450 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 219 | 27 | 10 |
| Malignant_lesions | 14 | 190 | 6 |
| body_of_stomach_normal | 2 | 0 | 203 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9319    0.8555    0.8921       256
     Malignant_lesions     0.8756    0.9048    0.8899       210
body_of_stomach_normal     0.9269    0.9902    0.9575       205

              accuracy                         0.9121       671
             macro avg     0.9115    0.9168    0.9132       671
          weighted avg     0.9128    0.9121    0.9114       671

```