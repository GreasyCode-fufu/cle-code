# convnext_fgc34_dsd2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc34_dsd2_seed42/convnext_fgc34_dsd2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9709 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9764 |
| Balanced Acc | 0.9649 |
| Macro-Sensitivity | 0.9649 |
| Macro-Specificity | 0.9866 |
| Macro-Precision | 0.9777 |
| AUC-PR (Macro) | 0.9927 |
| Cohen's Kappa | 0.9626 |
| MCC | 0.9628 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.40G |
| FLOPs | 8.80G |
| Params | 27.49M |
| Pure-Inference FPS | 705.4 (1.42 ms/张) |
| End-to-End FPS | 202.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9897 | 0.9655 | 0.9602 | 0.9747 | 0.9912 | 0.9971 | 0.9962 | 195 |
| Malignant_lesions | 0.9114 | 0.9943 | 0.9730 | 0.9412 | 0.9802 | 0.9920 | 0.9820 | 79 |
| body_of_stomach_normal | 0.9935 | 1.0000 | 1.0000 | 0.9967 | 0.9964 | 0.9999 | 0.9998 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=8 TN=224 FN=2
- **Malignant_lesions**: TP=72 FP=2 TN=346 FN=7
- **body_of_stomach_normal**: TP=152 FP=0 TN=274 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 2 | 0 |
| Malignant_lesions | 7 | 72 | 0 |
| body_of_stomach_normal | 1 | 0 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9602    0.9897    0.9747       195
     Malignant_lesions     0.9730    0.9114    0.9412        79
body_of_stomach_normal     1.0000    0.9935    0.9967       153

              accuracy                         0.9766       427
             macro avg     0.9777    0.9649    0.9709       427
          weighted avg     0.9768    0.9766    0.9764       427

```