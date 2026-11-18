# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/C/seed42/resnet34/resnet34_C_seed42_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8874 |
| Accuracy | 0.8852 |
| Macro-AUROC (OvR) | 0.9766 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8842 |
| Balanced Acc | 0.8945 |
| Macro-Sensitivity | 0.8945 |
| Macro-Specificity | 0.9436 |
| Macro-Precision | 0.8904 |
| AUC-PR (Macro) | 0.9578 |
| Cohen's Kappa | 0.8283 |
| MCC | 0.8334 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3796.7 (0.26 ms/张) |
| End-to-End FPS | 217.3 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7695 | 0.9735 | 0.9471 | 0.8491 | 0.8726 | 0.9685 | 0.9584 | 256 |
| Malignant_lesions | 0.9238 | 0.8872 | 0.7886 | 0.8509 | 0.9624 | 0.9649 | 0.9244 | 210 |
| body_of_stomach_normal | 0.9902 | 0.9700 | 0.9355 | 0.9621 | 0.9956 | 0.9963 | 0.9905 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=11 TN=404 FN=59
- **Malignant_lesions**: TP=194 FP=52 TN=409 FN=16
- **body_of_stomach_normal**: TP=203 FP=14 TN=452 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 51 | 8 |
| Malignant_lesions | 10 | 194 | 6 |
| body_of_stomach_normal | 1 | 1 | 203 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9471    0.7695    0.8491       256
     Malignant_lesions     0.7886    0.9238    0.8509       210
body_of_stomach_normal     0.9355    0.9902    0.9621       205

              accuracy                         0.8852       671
             macro avg     0.8904    0.8945    0.8874       671
          weighted avg     0.8940    0.8852    0.8842       671

```