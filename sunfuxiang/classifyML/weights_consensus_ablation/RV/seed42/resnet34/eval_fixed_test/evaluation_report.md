# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/RV/seed42/resnet34/resnet34_RV_seed42_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8994 |
| Accuracy | 0.8987 |
| Macro-AUROC (OvR) | 0.9761 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8979 |
| Balanced Acc | 0.9034 |
| Macro-Sensitivity | 0.9034 |
| Macro-Specificity | 0.9496 |
| Macro-Precision | 0.8978 |
| AUC-PR (Macro) | 0.9551 |
| Cohen's Kappa | 0.8478 |
| MCC | 0.8490 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3785.4 (0.26 ms/张) |
| End-to-End FPS | 243.5 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8438 | 0.9590 | 0.9270 | 0.8834 | 0.9087 | 0.9668 | 0.9556 | 256 |
| Malignant_lesions | 0.8762 | 0.9306 | 0.8519 | 0.8638 | 0.9429 | 0.9639 | 0.9149 | 210 |
| body_of_stomach_normal | 0.9902 | 0.9592 | 0.9144 | 0.9508 | 0.9955 | 0.9976 | 0.9948 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=216 FP=17 TN=398 FN=40
- **Malignant_lesions**: TP=184 FP=32 TN=429 FN=26
- **body_of_stomach_normal**: TP=203 FP=19 TN=447 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 216 | 31 | 9 |
| Malignant_lesions | 16 | 184 | 10 |
| body_of_stomach_normal | 1 | 1 | 203 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9270    0.8438    0.8834       256
     Malignant_lesions     0.8519    0.8762    0.8638       210
body_of_stomach_normal     0.9144    0.9902    0.9508       205

              accuracy                         0.8987       671
             macro avg     0.8978    0.9034    0.8994       671
          weighted avg     0.8997    0.8987    0.8979       671

```