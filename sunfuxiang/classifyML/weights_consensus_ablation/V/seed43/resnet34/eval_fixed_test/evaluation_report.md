# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/V/seed43/resnet34/resnet34_V_seed43_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8856 |
| Accuracy | 0.8852 |
| Macro-AUROC (OvR) | 0.9770 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8842 |
| Balanced Acc | 0.8864 |
| Macro-Sensitivity | 0.8864 |
| Macro-Specificity | 0.9416 |
| Macro-Precision | 0.8860 |
| AUC-PR (Macro) | 0.9605 |
| Cohen's Kappa | 0.8269 |
| MCC | 0.8274 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3798.5 (0.26 ms/张) |
| End-to-End FPS | 232.8 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8789 | 0.9157 | 0.8654 | 0.8721 | 0.9246 | 0.9677 | 0.9554 | 256 |
| Malignant_lesions | 0.8048 | 0.9436 | 0.8667 | 0.8346 | 0.9139 | 0.9676 | 0.9365 | 210 |
| body_of_stomach_normal | 0.9756 | 0.9657 | 0.9259 | 0.9501 | 0.9890 | 0.9957 | 0.9896 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=225 FP=35 TN=380 FN=31
- **Malignant_lesions**: TP=169 FP=26 TN=435 FN=41
- **body_of_stomach_normal**: TP=200 FP=16 TN=450 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 225 | 22 | 9 |
| Malignant_lesions | 34 | 169 | 7 |
| body_of_stomach_normal | 1 | 4 | 200 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8654    0.8789    0.8721       256
     Malignant_lesions     0.8667    0.8048    0.8346       210
body_of_stomach_normal     0.9259    0.9756    0.9501       205

              accuracy                         0.8852       671
             macro avg     0.8860    0.8864    0.8856       671
          weighted avg     0.8843    0.8852    0.8842       671

```