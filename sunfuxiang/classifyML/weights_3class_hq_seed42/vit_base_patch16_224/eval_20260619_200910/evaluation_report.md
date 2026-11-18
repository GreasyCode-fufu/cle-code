# vit_base_patch16_224 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_hq_seed42/vit_base_patch16_224/vit_base_patch16_224_3class_hq_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8779 |
| Accuracy | 0.8763 |
| Macro-AUROC (OvR) | 0.9771 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8751 |
| Balanced Acc | 0.8853 |
| Macro-Sensitivity | 0.8853 |
| Macro-Specificity | 0.9392 |
| Macro-Precision | 0.8811 |
| AUC-PR (Macro) | 0.9618 |
| Cohen's Kappa | 0.8149 |
| MCC | 0.8202 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 870.3 (1.15 ms/张) |
| End-to-End FPS | 210.9 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7617 | 0.9711 | 0.9420 | 0.8423 | 0.8685 | 0.9663 | 0.9529 | 256 |
| Malignant_lesions | 0.9333 | 0.8894 | 0.7935 | 0.8578 | 0.9670 | 0.9715 | 0.9472 | 210 |
| body_of_stomach_normal | 0.9610 | 0.9571 | 0.9078 | 0.9336 | 0.9824 | 0.9936 | 0.9854 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=12 TN=403 FN=61
- **Malignant_lesions**: TP=196 FP=51 TN=410 FN=14
- **body_of_stomach_normal**: TP=197 FP=20 TN=446 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 47 | 14 |
| Malignant_lesions | 8 | 196 | 6 |
| body_of_stomach_normal | 4 | 4 | 197 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9420    0.7617    0.8423       256
     Malignant_lesions     0.7935    0.9333    0.8578       210
body_of_stomach_normal     0.9078    0.9610    0.9336       205

              accuracy                         0.8763       671
             macro avg     0.8811    0.8853    0.8779       671
          weighted avg     0.8851    0.8763    0.8751       671

```