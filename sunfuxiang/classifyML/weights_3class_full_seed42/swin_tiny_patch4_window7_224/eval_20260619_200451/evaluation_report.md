# swin_tiny_patch4_window7_224 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed42/swin_tiny_patch4_window7_224/swin_tiny_patch4_window7_224_3class_full_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8754 |
| Accuracy | 0.8718 |
| Macro-AUROC (OvR) | 0.9674 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8729 |
| Balanced Acc | 0.8784 |
| Macro-Sensitivity | 0.8784 |
| Macro-Specificity | 0.9363 |
| Macro-Precision | 0.8799 |
| AUC-PR (Macro) | 0.9417 |
| Cohen's Kappa | 0.8077 |
| MCC | 0.8117 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.92G |
| Params | 27.94M |
| Pure-Inference FPS | 1527.8 (0.65 ms/张) |
| End-to-End FPS | 212.1 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7891 | 0.9542 | 0.9140 | 0.8470 | 0.8800 | 0.9542 | 0.9380 | 256 |
| Malignant_lesions | 0.9095 | 0.8698 | 0.7610 | 0.8286 | 0.9548 | 0.9527 | 0.8979 | 210 |
| body_of_stomach_normal | 0.9366 | 0.9850 | 0.9648 | 0.9505 | 0.9725 | 0.9952 | 0.9892 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=19 TN=396 FN=54
- **Malignant_lesions**: TP=191 FP=60 TN=401 FN=19
- **body_of_stomach_normal**: TP=192 FP=7 TN=459 FN=13

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 49 | 5 |
| Malignant_lesions | 17 | 191 | 2 |
| body_of_stomach_normal | 2 | 11 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9140    0.7891    0.8470       256
     Malignant_lesions     0.7610    0.9095    0.8286       210
body_of_stomach_normal     0.9648    0.9366    0.9505       205

              accuracy                         0.8718       671
             macro avg     0.8799    0.8784    0.8754       671
          weighted avg     0.8816    0.8718    0.8729       671

```