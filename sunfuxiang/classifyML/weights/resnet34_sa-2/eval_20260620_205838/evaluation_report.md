# resnet34_sa 评估报告

- 权重: `./weights/resnet34_sa-2/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8805 |
| Accuracy | 0.8778 |
| Macro-AUROC (OvR) | 0.9723 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8771 |
| Balanced Acc | 0.8875 |
| Macro-Sensitivity | 0.8875 |
| Macro-Specificity | 0.9398 |
| Macro-Precision | 0.8868 |
| AUC-PR (Macro) | 0.9503 |
| Cohen's Kappa | 0.8172 |
| MCC | 0.8240 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3724.7 (0.27 ms/张) |
| End-to-End FPS | 224.4 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7539 | 0.9711 | 0.9415 | 0.8373 | 0.8648 | 0.9558 | 0.9439 | 256 |
| Malignant_lesions | 0.9524 | 0.8720 | 0.7722 | 0.8529 | 0.9757 | 0.9669 | 0.9179 | 210 |
| body_of_stomach_normal | 0.9561 | 0.9764 | 0.9469 | 0.9515 | 0.9806 | 0.9943 | 0.9891 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=12 TN=403 FN=63
- **Malignant_lesions**: TP=200 FP=59 TN=402 FN=10
- **body_of_stomach_normal**: TP=196 FP=11 TN=455 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 54 | 9 |
| Malignant_lesions | 8 | 200 | 2 |
| body_of_stomach_normal | 4 | 5 | 196 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9415    0.7539    0.8373       256
     Malignant_lesions     0.7722    0.9524    0.8529       210
body_of_stomach_normal     0.9469    0.9561    0.9515       205

              accuracy                         0.8778       671
             macro avg     0.8868    0.8875    0.8805       671
          weighted avg     0.8901    0.8778    0.8771       671

```