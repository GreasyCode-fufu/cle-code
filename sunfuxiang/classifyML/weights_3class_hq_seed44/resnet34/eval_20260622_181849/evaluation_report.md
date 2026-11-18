# resnet34 评估报告

- 权重: `weights_3class_hq_seed44/resnet34/resnet34_3class_hq_seed44_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9089 |
| Accuracy | 0.9076 |
| Macro-AUROC (OvR) | 0.9809 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9075 |
| Balanced Acc | 0.9095 |
| Macro-Sensitivity | 0.9095 |
| Macro-Specificity | 0.9533 |
| Macro-Precision | 0.9082 |
| AUC-PR (Macro) | 0.9680 |
| Cohen's Kappa | 0.8608 |
| MCC | 0.8608 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3740.8 (0.27 ms/张) |
| End-to-End FPS | 248.7 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8867 | 0.9398 | 0.9008 | 0.8937 | 0.9308 | 0.9739 | 0.9624 | 256 |
| Malignant_lesions | 0.8810 | 0.9436 | 0.8768 | 0.8789 | 0.9457 | 0.9725 | 0.9490 | 210 |
| body_of_stomach_normal | 0.9610 | 0.9764 | 0.9471 | 0.9540 | 0.9827 | 0.9964 | 0.9926 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=227 FP=25 TN=390 FN=29
- **Malignant_lesions**: TP=185 FP=26 TN=435 FN=25
- **body_of_stomach_normal**: TP=197 FP=11 TN=455 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 227 | 20 | 9 |
| Malignant_lesions | 23 | 185 | 2 |
| body_of_stomach_normal | 2 | 6 | 197 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9008    0.8867    0.8937       256
     Malignant_lesions     0.8768    0.8810    0.8789       210
body_of_stomach_normal     0.9471    0.9610    0.9540       205

              accuracy                         0.9076       671
             macro avg     0.9082    0.9095    0.9089       671
          weighted avg     0.9074    0.9076    0.9075       671

```