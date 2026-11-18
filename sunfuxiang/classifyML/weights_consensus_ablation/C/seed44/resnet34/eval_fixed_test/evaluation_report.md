# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/C/seed44/resnet34/resnet34_C_seed44_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8988 |
| Accuracy | 0.8972 |
| Macro-AUROC (OvR) | 0.9730 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8969 |
| Balanced Acc | 0.8994 |
| Macro-Sensitivity | 0.8994 |
| Macro-Specificity | 0.9478 |
| Macro-Precision | 0.8982 |
| AUC-PR (Macro) | 0.9537 |
| Cohen's Kappa | 0.8450 |
| MCC | 0.8450 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3817.8 (0.26 ms/张) |
| End-to-End FPS | 230.6 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8750 | 0.9277 | 0.8819 | 0.8784 | 0.9233 | 0.9640 | 0.9492 | 256 |
| Malignant_lesions | 0.8476 | 0.9349 | 0.8558 | 0.8517 | 0.9309 | 0.9583 | 0.9202 | 210 |
| body_of_stomach_normal | 0.9756 | 0.9807 | 0.9569 | 0.9662 | 0.9892 | 0.9967 | 0.9919 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=224 FP=30 TN=385 FN=32
- **Malignant_lesions**: TP=178 FP=30 TN=431 FN=32
- **body_of_stomach_normal**: TP=200 FP=9 TN=457 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 224 | 27 | 5 |
| Malignant_lesions | 28 | 178 | 4 |
| body_of_stomach_normal | 2 | 3 | 200 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8819    0.8750    0.8784       256
     Malignant_lesions     0.8558    0.8476    0.8517       210
body_of_stomach_normal     0.9569    0.9756    0.9662       205

              accuracy                         0.8972       671
             macro avg     0.8982    0.8994    0.8988       671
          weighted avg     0.8966    0.8972    0.8969       671

```