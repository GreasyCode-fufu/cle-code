# mbv3_l2sp_a001 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_l2sp_a001_seed44/mbv3_l2sp_a001/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9218 |
| Accuracy | 0.9236 |
| Macro-AUROC (OvR) | 0.9922 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9226 |
| Balanced Acc | 0.9190 |
| Macro-Sensitivity | 0.9190 |
| Macro-Specificity | 0.9607 |
| Macro-Precision | 0.9324 |
| AUC-PR (Macro) | 0.9852 |
| Cohen's Kappa | 0.8847 |
| MCC | 0.8881 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3769.7 (0.27 ms/张) |
| End-to-End FPS | 149.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9076 | 0.8553 | 0.9074 | 0.9795 | 0.9857 | 0.9760 | 208 |
| Malignant_lesions | 0.8011 | 0.9900 | 0.9724 | 0.8785 | 0.9188 | 0.9914 | 0.9804 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9844 | 0.9694 | 0.9794 | 0.9947 | 0.9996 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=34 TN=334 FN=7
- **Malignant_lesions**: TP=141 FP=4 TN=396 FN=35
- **body_of_stomach_normal**: TP=190 FP=6 TN=378 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 4 | 3 |
| Malignant_lesions | 32 | 141 | 3 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8553    0.9663    0.9074       208
     Malignant_lesions     0.9724    0.8011    0.8785       176
body_of_stomach_normal     0.9694    0.9896    0.9794       192

              accuracy                         0.9236       576
             macro avg     0.9324    0.9190    0.9218       576
          weighted avg     0.9291    0.9236    0.9226       576

```