# resnet34_degconv_cpca 评估报告

- 权重: `./weights/resnet34_degconv_cpca-2/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9022 |
| Accuracy | 0.9001 |
| Macro-AUROC (OvR) | 0.9816 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9007 |
| Balanced Acc | 0.9049 |
| Macro-Sensitivity | 0.9049 |
| Macro-Specificity | 0.9505 |
| Macro-Precision | 0.9039 |
| AUC-PR (Macro) | 0.9682 |
| Cohen's Kappa | 0.8500 |
| MCC | 0.8524 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.87G |
| FLOPs | 7.74G |
| Params | 21.95M |
| Pure-Inference FPS | 975.8 (1.02 ms/张) |
| End-to-End FPS | 203.7 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8398 | 0.9663 | 0.9389 | 0.8866 | 0.9072 | 0.9751 | 0.9662 | 256 |
| Malignant_lesions | 0.9286 | 0.9024 | 0.8125 | 0.8667 | 0.9652 | 0.9732 | 0.9469 | 210 |
| body_of_stomach_normal | 0.9463 | 0.9828 | 0.9604 | 0.9533 | 0.9765 | 0.9965 | 0.9916 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=215 FP=14 TN=401 FN=41
- **Malignant_lesions**: TP=195 FP=45 TN=416 FN=15
- **body_of_stomach_normal**: TP=194 FP=8 TN=458 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 215 | 36 | 5 |
| Malignant_lesions | 12 | 195 | 3 |
| body_of_stomach_normal | 2 | 9 | 194 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9389    0.8398    0.8866       256
     Malignant_lesions     0.8125    0.9286    0.8667       210
body_of_stomach_normal     0.9604    0.9463    0.9533       205

              accuracy                         0.9001       671
             macro avg     0.9039    0.9049    0.9022       671
          weighted avg     0.9059    0.9001    0.9007       671

```