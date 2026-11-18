# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-12/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8831 |
| Accuracy | 0.8808 |
| Macro-AUROC (OvR) | 0.9750 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8799 |
| Balanced Acc | 0.8907 |
| Macro-Sensitivity | 0.8907 |
| Macro-Specificity | 0.9415 |
| Macro-Precision | 0.8899 |
| AUC-PR (Macro) | 0.9526 |
| Cohen's Kappa | 0.8217 |
| MCC | 0.8291 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.71G |
| FLOPs | 9.42G |
| Params | 24.58M |
| Pure-Inference FPS | 3070.1 (0.33 ms/张) |
| End-to-End FPS | 247.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7539 | 0.9783 | 0.9554 | 0.8428 | 0.8657 | 0.9621 | 0.9527 | 256 |
| Malignant_lesions | 0.9571 | 0.8742 | 0.7761 | 0.8571 | 0.9782 | 0.9673 | 0.9151 | 210 |
| body_of_stomach_normal | 0.9610 | 0.9721 | 0.9381 | 0.9494 | 0.9826 | 0.9957 | 0.9901 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=9 TN=406 FN=63
- **Malignant_lesions**: TP=201 FP=58 TN=403 FN=9
- **body_of_stomach_normal**: TP=197 FP=13 TN=453 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 53 | 10 |
| Malignant_lesions | 6 | 201 | 3 |
| body_of_stomach_normal | 3 | 5 | 197 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9554    0.7539    0.8428       256
     Malignant_lesions     0.7761    0.9571    0.8571       210
body_of_stomach_normal     0.9381    0.9610    0.9494       205

              accuracy                         0.8808       671
             macro avg     0.8899    0.8907    0.8831       671
          weighted avg     0.8940    0.8808    0.8799       671

```