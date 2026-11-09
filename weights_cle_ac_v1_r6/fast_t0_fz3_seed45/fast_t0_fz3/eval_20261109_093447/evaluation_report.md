# fast_t0_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_seed45/fast_t0_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9403 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9916 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9407 |
| Balanced Acc | 0.9393 |
| Macro-Sensitivity | 0.9393 |
| Macro-Specificity | 0.9701 |
| Macro-Precision | 0.9422 |
| AUC-PR (Macro) | 0.9830 |
| Cohen's Kappa | 0.9111 |
| MCC | 0.9115 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 7124.0 (0.14 ms/张) |
| End-to-End FPS | 220.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9484 | 0.9120 | 0.9292 | 0.9694 | 0.9872 | 0.9805 | 208 |
| Malignant_lesions | 0.8864 | 0.9775 | 0.9455 | 0.9150 | 0.9513 | 0.9886 | 0.9703 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9844 | 0.9692 | 0.9767 | 0.9921 | 0.9991 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=19 TN=349 FN=11
- **Malignant_lesions**: TP=156 FP=9 TN=391 FN=20
- **body_of_stomach_normal**: TP=189 FP=6 TN=378 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 16 | 156 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9120    0.9471    0.9292       208
     Malignant_lesions     0.9455    0.8864    0.9150       176
body_of_stomach_normal     0.9692    0.9844    0.9767       192

              accuracy                         0.9410       576
             macro avg     0.9422    0.9393    0.9403       576
          weighted avg     0.9413    0.9410    0.9407       576

```