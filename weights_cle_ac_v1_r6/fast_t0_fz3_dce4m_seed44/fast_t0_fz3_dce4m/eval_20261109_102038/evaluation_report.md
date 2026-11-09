# fast_t0_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_dce4m_seed44/fast_t0_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9682 |
| Accuracy | 0.9742 |
| Macro-AUROC (OvR) | 0.9973 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9741 |
| Balanced Acc | 0.9661 |
| Macro-Sensitivity | 0.9661 |
| Macro-Specificity | 0.9863 |
| Macro-Precision | 0.9703 |
| AUC-PR (Macro) | 0.9914 |
| Cohen's Kappa | 0.9590 |
| MCC | 0.9590 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 5994.1 (0.17 ms/张) |
| End-to-End FPS | 8.3 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9741 | 0.9694 | 0.9719 | 0.9784 | 0.9976 | 0.9971 | 195 |
| Malignant_lesions | 0.9241 | 0.9885 | 0.9481 | 0.9359 | 0.9829 | 0.9943 | 0.9771 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=6 TN=226 FN=5
- **Malignant_lesions**: TP=73 FP=4 TN=344 FN=6
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 4 | 1 |
| Malignant_lesions | 6 | 73 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9694    0.9744    0.9719       195
     Malignant_lesions     0.9481    0.9241    0.9359        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9742       427
             macro avg     0.9703    0.9661    0.9682       427
          weighted avg     0.9741    0.9742    0.9741       427

```