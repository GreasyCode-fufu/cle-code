# swin_tiny_patch4_window7_224 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_hq_seed43/swin_tiny_patch4_window7_224/swin_tiny_patch4_window7_224_3class_hq_seed43_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9108 |
| Accuracy | 0.9091 |
| Macro-AUROC (OvR) | 0.9824 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9088 |
| Balanced Acc | 0.9126 |
| Macro-Sensitivity | 0.9126 |
| Macro-Specificity | 0.9542 |
| Macro-Precision | 0.9095 |
| AUC-PR (Macro) | 0.9696 |
| Cohen's Kappa | 0.8632 |
| MCC | 0.8635 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.92G |
| Params | 27.94M |
| Pure-Inference FPS | 1522.5 (0.66 ms/张) |
| End-to-End FPS | 222.7 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8672 | 0.9470 | 0.9098 | 0.8880 | 0.9204 | 0.9765 | 0.9678 | 256 |
| Malignant_lesions | 0.9000 | 0.9393 | 0.8710 | 0.8852 | 0.9537 | 0.9753 | 0.9515 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9764 | 0.9476 | 0.9590 | 0.9870 | 0.9955 | 0.9894 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=222 FP=22 TN=393 FN=34
- **Malignant_lesions**: TP=189 FP=28 TN=433 FN=21
- **body_of_stomach_normal**: TP=199 FP=11 TN=455 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 222 | 25 | 9 |
| Malignant_lesions | 19 | 189 | 2 |
| body_of_stomach_normal | 3 | 3 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9098    0.8672    0.8880       256
     Malignant_lesions     0.8710    0.9000    0.8852       210
body_of_stomach_normal     0.9476    0.9707    0.9590       205

              accuracy                         0.9091       671
             macro avg     0.9095    0.9126    0.9108       671
          weighted avg     0.9092    0.9091    0.9088       671

```