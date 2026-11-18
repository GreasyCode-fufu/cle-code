# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-10/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8866 |
| Accuracy | 0.8852 |
| Macro-AUROC (OvR) | 0.9626 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8839 |
| Balanced Acc | 0.8931 |
| Macro-Sensitivity | 0.8931 |
| Macro-Specificity | 0.9433 |
| Macro-Precision | 0.8865 |
| AUC-PR (Macro) | 0.9372 |
| Cohen's Kappa | 0.8280 |
| MCC | 0.8312 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.71G |
| FLOPs | 9.42G |
| Params | 24.58M |
| Pure-Inference FPS | 3039.7 (0.33 ms/张) |
| End-to-End FPS | 240.8 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7891 | 0.9639 | 0.9309 | 0.8541 | 0.8811 | 0.9414 | 0.9249 | 256 |
| Malignant_lesions | 0.9000 | 0.9089 | 0.8182 | 0.8571 | 0.9523 | 0.9517 | 0.9038 | 210 |
| body_of_stomach_normal | 0.9902 | 0.9571 | 0.9103 | 0.9486 | 0.9955 | 0.9947 | 0.9830 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=15 TN=400 FN=54
- **Malignant_lesions**: TP=189 FP=42 TN=419 FN=21
- **body_of_stomach_normal**: TP=203 FP=20 TN=446 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 40 | 14 |
| Malignant_lesions | 15 | 189 | 6 |
| body_of_stomach_normal | 0 | 2 | 203 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9309    0.7891    0.8541       256
     Malignant_lesions     0.8182    0.9000    0.8571       210
body_of_stomach_normal     0.9103    0.9902    0.9486       205

              accuracy                         0.8852       671
             macro avg     0.8865    0.8931    0.8866       671
          weighted avg     0.8893    0.8852    0.8839       671

```