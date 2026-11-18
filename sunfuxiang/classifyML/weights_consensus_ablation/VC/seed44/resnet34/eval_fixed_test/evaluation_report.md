# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/VC/seed44/resnet34/resnet34_VC_seed44_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8988 |
| Accuracy | 0.8972 |
| Macro-AUROC (OvR) | 0.9792 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8970 |
| Balanced Acc | 0.9032 |
| Macro-Sensitivity | 0.9032 |
| Macro-Specificity | 0.9492 |
| Macro-Precision | 0.8995 |
| AUC-PR (Macro) | 0.9625 |
| Cohen's Kappa | 0.8457 |
| MCC | 0.8484 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3768.2 (0.27 ms/张) |
| End-to-End FPS | 221.7 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8203 | 0.9687 | 0.9417 | 0.8768 | 0.8973 | 0.9724 | 0.9629 | 256 |
| Malignant_lesions | 0.9333 | 0.9089 | 0.8235 | 0.8750 | 0.9677 | 0.9716 | 0.9374 | 210 |
| body_of_stomach_normal | 0.9561 | 0.9700 | 0.9333 | 0.9446 | 0.9805 | 0.9936 | 0.9871 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=210 FP=13 TN=402 FN=46
- **Malignant_lesions**: TP=196 FP=42 TN=419 FN=14
- **body_of_stomach_normal**: TP=196 FP=14 TN=452 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 210 | 35 | 11 |
| Malignant_lesions | 11 | 196 | 3 |
| body_of_stomach_normal | 2 | 7 | 196 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9417    0.8203    0.8768       256
     Malignant_lesions     0.8235    0.9333    0.8750       210
body_of_stomach_normal     0.9333    0.9561    0.9446       205

              accuracy                         0.8972       671
             macro avg     0.8995    0.9032    0.8988       671
          weighted avg     0.9022    0.8972    0.8970       671

```