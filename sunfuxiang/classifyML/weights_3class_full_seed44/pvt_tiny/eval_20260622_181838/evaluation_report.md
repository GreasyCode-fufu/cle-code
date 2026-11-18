# pvt_tiny 评估报告

- 权重: `weights_3class_full_seed44/pvt_tiny/pvt_tiny_3class_full_seed44_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8864 |
| Accuracy | 0.8852 |
| Macro-AUROC (OvR) | 0.9743 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8849 |
| Balanced Acc | 0.8881 |
| Macro-Sensitivity | 0.8881 |
| Macro-Specificity | 0.9422 |
| Macro-Precision | 0.8852 |
| AUC-PR (Macro) | 0.9566 |
| Cohen's Kappa | 0.8272 |
| MCC | 0.8274 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2912.7 (0.34 ms/张) |
| End-to-End FPS | 228.4 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8555 | 0.9325 | 0.8866 | 0.8708 | 0.9127 | 0.9664 | 0.9551 | 256 |
| Malignant_lesions | 0.8429 | 0.9241 | 0.8349 | 0.8389 | 0.9281 | 0.9633 | 0.9353 | 210 |
| body_of_stomach_normal | 0.9659 | 0.9700 | 0.9340 | 0.9496 | 0.9847 | 0.9931 | 0.9795 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=219 FP=28 TN=387 FN=37
- **Malignant_lesions**: TP=177 FP=35 TN=426 FN=33
- **body_of_stomach_normal**: TP=198 FP=14 TN=452 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 219 | 29 | 8 |
| Malignant_lesions | 27 | 177 | 6 |
| body_of_stomach_normal | 1 | 6 | 198 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8866    0.8555    0.8708       256
     Malignant_lesions     0.8349    0.8429    0.8389       210
body_of_stomach_normal     0.9340    0.9659    0.9496       205

              accuracy                         0.8852       671
             macro avg     0.8852    0.8881    0.8864       671
          weighted avg     0.8849    0.8852    0.8849       671

```