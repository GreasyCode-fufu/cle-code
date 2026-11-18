# LightGBM 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8754 |
| Accuracy | 0.8763 |
| Macro-AUROC (OvR) | 0.9265 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8747 |
| Balanced Acc | 0.8784 |
| Macro-Sensitivity | 0.8784 |
| Macro-Specificity | 0.9378 |
| Macro-Precision | 0.8773 |
| AUC-PR (Macro) | 0.8181 |
| Cohen's Kappa | 0.8137 |
| MCC | 0.8158 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 373.55M |
| FLOPs | 747.10M |
| Params | 3.70M |
| Pure-Inference FPS | 382804.4 (0.00 ms/张) |
| End-to-End FPS | 774.1 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8594 | 0.9277 | 0.8800 | 0.8696 | 0.9145 | 0.9041 | 0.8129 | 256 |
| Malignant_lesions | 0.7905 | 0.9566 | 0.8925 | 0.8384 | 0.9093 | 0.9297 | 0.8524 | 210 |
| body_of_stomach_normal | 0.9854 | 0.9292 | 0.8596 | 0.9182 | 0.9931 | 0.9455 | 0.7891 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=220 FP=30 TN=385 FN=36
- **Malignant_lesions**: TP=166 FP=20 TN=441 FN=44
- **body_of_stomach_normal**: TP=202 FP=33 TN=433 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 220 | 19 | 17 |
| Malignant_lesions | 28 | 166 | 16 |
| body_of_stomach_normal | 2 | 1 | 202 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8800    0.8594    0.8696       256
     Malignant_lesions     0.8925    0.7905    0.8384       210
body_of_stomach_normal     0.8596    0.9854    0.9182       205

              accuracy                         0.8763       671
             macro avg     0.8773    0.8784    0.8754       671
          weighted avg     0.8777    0.8763    0.8747       671

```