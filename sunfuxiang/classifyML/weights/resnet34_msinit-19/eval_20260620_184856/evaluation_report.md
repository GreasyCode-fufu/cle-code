# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-19/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8885 |
| Accuracy | 0.8882 |
| Macro-AUROC (OvR) | 0.9748 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8872 |
| Balanced Acc | 0.8893 |
| Macro-Sensitivity | 0.8893 |
| Macro-Specificity | 0.9432 |
| Macro-Precision | 0.8890 |
| AUC-PR (Macro) | 0.9555 |
| Cohen's Kappa | 0.8314 |
| MCC | 0.8319 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.31G |
| FLOPs | 8.63G |
| Params | 21.86M |
| Pure-Inference FPS | 2715.0 (0.37 ms/张) |
| End-to-End FPS | 242.9 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8828 | 0.9181 | 0.8692 | 0.8760 | 0.9270 | 0.9619 | 0.9491 | 256 |
| Malignant_lesions | 0.8095 | 0.9458 | 0.8718 | 0.8395 | 0.9160 | 0.9661 | 0.9265 | 210 |
| body_of_stomach_normal | 0.9756 | 0.9657 | 0.9259 | 0.9501 | 0.9890 | 0.9963 | 0.9910 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=226 FP=34 TN=381 FN=30
- **Malignant_lesions**: TP=170 FP=25 TN=436 FN=40
- **body_of_stomach_normal**: TP=200 FP=16 TN=450 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 226 | 22 | 8 |
| Malignant_lesions | 32 | 170 | 8 |
| body_of_stomach_normal | 2 | 3 | 200 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8692    0.8828    0.8760       256
     Malignant_lesions     0.8718    0.8095    0.8395       210
body_of_stomach_normal     0.9259    0.9756    0.9501       205

              accuracy                         0.8882       671
             macro avg     0.8890    0.8893    0.8885       671
          weighted avg     0.8874    0.8882    0.8872       671

```