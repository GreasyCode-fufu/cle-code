# vit_base_patch16_224 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed44/vit_base_patch16_224/vit_base_patch16_224_3class_full_seed44_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8758 |
| Accuracy | 0.8718 |
| Macro-AUROC (OvR) | 0.9735 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8737 |
| Balanced Acc | 0.8776 |
| Macro-Sensitivity | 0.8776 |
| Macro-Specificity | 0.9363 |
| Macro-Precision | 0.8839 |
| AUC-PR (Macro) | 0.9517 |
| Cohen's Kappa | 0.8077 |
| MCC | 0.8127 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 866.6 (1.15 ms/张) |
| End-to-End FPS | 212.9 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7969 | 0.9542 | 0.9148 | 0.8518 | 0.8839 | 0.9622 | 0.9452 | 256 |
| Malignant_lesions | 0.9286 | 0.8612 | 0.7529 | 0.8316 | 0.9636 | 0.9636 | 0.9199 | 210 |
| body_of_stomach_normal | 0.9073 | 0.9936 | 0.9841 | 0.9442 | 0.9606 | 0.9947 | 0.9901 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=19 TN=396 FN=52
- **Malignant_lesions**: TP=195 FP=64 TN=397 FN=15
- **body_of_stomach_normal**: TP=186 FP=3 TN=463 FN=19

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 51 | 1 |
| Malignant_lesions | 13 | 195 | 2 |
| body_of_stomach_normal | 6 | 13 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9148    0.7969    0.8518       256
     Malignant_lesions     0.7529    0.9286    0.8316       210
body_of_stomach_normal     0.9841    0.9073    0.9442       205

              accuracy                         0.8718       671
             macro avg     0.8839    0.8776    0.8758       671
          weighted avg     0.8853    0.8718    0.8737       671

```