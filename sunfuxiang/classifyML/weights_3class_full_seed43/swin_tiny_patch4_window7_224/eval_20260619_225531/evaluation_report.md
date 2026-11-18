# swin_tiny_patch4_window7_224 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed43/swin_tiny_patch4_window7_224/swin_tiny_patch4_window7_224_3class_full_seed43_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8784 |
| Accuracy | 0.8778 |
| Macro-AUROC (OvR) | 0.9733 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8765 |
| Balanced Acc | 0.8843 |
| Macro-Sensitivity | 0.8843 |
| Macro-Specificity | 0.9395 |
| Macro-Precision | 0.8775 |
| AUC-PR (Macro) | 0.9572 |
| Cohen's Kappa | 0.8167 |
| MCC | 0.8192 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.92G |
| Params | 27.94M |
| Pure-Inference FPS | 1523.0 (0.66 ms/张) |
| End-to-End FPS | 228.3 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8008 | 0.9566 | 0.9193 | 0.8559 | 0.8862 | 0.9592 | 0.9480 | 256 |
| Malignant_lesions | 0.8667 | 0.9219 | 0.8349 | 0.8505 | 0.9382 | 0.9651 | 0.9347 | 210 |
| body_of_stomach_normal | 0.9854 | 0.9399 | 0.8783 | 0.9287 | 0.9932 | 0.9955 | 0.9889 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=205 FP=18 TN=397 FN=51
- **Malignant_lesions**: TP=182 FP=36 TN=425 FN=28
- **body_of_stomach_normal**: TP=202 FP=28 TN=438 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 205 | 34 | 17 |
| Malignant_lesions | 17 | 182 | 11 |
| body_of_stomach_normal | 1 | 2 | 202 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9193    0.8008    0.8559       256
     Malignant_lesions     0.8349    0.8667    0.8505       210
body_of_stomach_normal     0.8783    0.9854    0.9287       205

              accuracy                         0.8778       671
             macro avg     0.8775    0.8843    0.8784       671
          weighted avg     0.8803    0.8778    0.8765       671

```