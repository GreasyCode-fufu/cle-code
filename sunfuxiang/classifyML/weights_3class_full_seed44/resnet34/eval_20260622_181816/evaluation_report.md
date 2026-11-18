# resnet34 评估报告

- 权重: `weights_3class_full_seed44/resnet34/resnet34_3class_full_seed44_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8737 |
| Accuracy | 0.8718 |
| Macro-AUROC (OvR) | 0.9662 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8709 |
| Balanced Acc | 0.8771 |
| Macro-Sensitivity | 0.8771 |
| Macro-Specificity | 0.9355 |
| Macro-Precision | 0.8717 |
| AUC-PR (Macro) | 0.9361 |
| Cohen's Kappa | 0.8073 |
| MCC | 0.8080 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3816.7 (0.26 ms/张) |
| End-to-End FPS | 245.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8125 | 0.9277 | 0.8739 | 0.8421 | 0.8891 | 0.9515 | 0.9355 | 256 |
| Malignant_lesions | 0.8286 | 0.9089 | 0.8056 | 0.8169 | 0.9209 | 0.9502 | 0.8824 | 210 |
| body_of_stomach_normal | 0.9902 | 0.9700 | 0.9355 | 0.9621 | 0.9956 | 0.9968 | 0.9903 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=208 FP=30 TN=385 FN=48
- **Malignant_lesions**: TP=174 FP=42 TN=419 FN=36
- **body_of_stomach_normal**: TP=203 FP=14 TN=452 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 208 | 40 | 8 |
| Malignant_lesions | 30 | 174 | 6 |
| body_of_stomach_normal | 0 | 2 | 203 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8739    0.8125    0.8421       256
     Malignant_lesions     0.8056    0.8286    0.8169       210
body_of_stomach_normal     0.9355    0.9902    0.9621       205

              accuracy                         0.8718       671
             macro avg     0.8717    0.8771    0.8737       671
          weighted avg     0.8713    0.8718    0.8709       671

```