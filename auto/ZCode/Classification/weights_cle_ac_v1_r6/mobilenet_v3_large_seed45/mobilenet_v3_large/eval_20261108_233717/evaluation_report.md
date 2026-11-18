# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mobilenet_v3_large_seed45/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9558 |
| Accuracy | 0.9649 |
| Macro-AUROC (OvR) | 0.9958 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9650 |
| Balanced Acc | 0.9593 |
| Macro-Sensitivity | 0.9593 |
| Macro-Specificity | 0.9830 |
| Macro-Precision | 0.9525 |
| AUC-PR (Macro) | 0.9870 |
| Cohen's Kappa | 0.9443 |
| MCC | 0.9444 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 3819.9 (0.26 ms/张) |
| End-to-End FPS | 270.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9538 | 0.9784 | 0.9738 | 0.9637 | 0.9619 | 0.9961 | 0.9955 | 195 |
| Malignant_lesions | 0.9241 | 0.9741 | 0.8902 | 0.9068 | 0.9826 | 0.9914 | 0.9655 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=186 FP=5 TN=227 FN=9
- **Malignant_lesions**: TP=73 FP=9 TN=339 FN=6
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 186 | 9 | 0 |
| Malignant_lesions | 5 | 73 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9738    0.9538    0.9637       195
     Malignant_lesions     0.8902    0.9241    0.9068        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9649       427
             macro avg     0.9525    0.9593    0.9558       427
          weighted avg     0.9654    0.9649    0.9650       427

```