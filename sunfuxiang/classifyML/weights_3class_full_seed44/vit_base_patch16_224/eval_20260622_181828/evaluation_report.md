# vit_base_patch16_224 评估报告

- 权重: `weights_3class_full_seed44/vit_base_patch16_224/vit_base_patch16_224_3class_full_seed44_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8744 |
| Accuracy | 0.8703 |
| Macro-AUROC (OvR) | 0.9735 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8722 |
| Balanced Acc | 0.8763 |
| Macro-Sensitivity | 0.8763 |
| Macro-Specificity | 0.9356 |
| Macro-Precision | 0.8828 |
| AUC-PR (Macro) | 0.9517 |
| Cohen's Kappa | 0.8055 |
| MCC | 0.8107 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 869.4 (1.15 ms/张) |
| End-to-End FPS | 189.1 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7930 | 0.9542 | 0.9144 | 0.8494 | 0.8820 | 0.9622 | 0.9452 | 256 |
| Malignant_lesions | 0.9286 | 0.8590 | 0.7500 | 0.8298 | 0.9635 | 0.9637 | 0.9200 | 210 |
| body_of_stomach_normal | 0.9073 | 0.9936 | 0.9841 | 0.9442 | 0.9606 | 0.9947 | 0.9900 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=19 TN=396 FN=53
- **Malignant_lesions**: TP=195 FP=65 TN=396 FN=15
- **body_of_stomach_normal**: TP=186 FP=3 TN=463 FN=19

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 52 | 1 |
| Malignant_lesions | 13 | 195 | 2 |
| body_of_stomach_normal | 6 | 13 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9144    0.7930    0.8494       256
     Malignant_lesions     0.7500    0.9286    0.8298       210
body_of_stomach_normal     0.9841    0.9073    0.9442       205

              accuracy                         0.8703       671
             macro avg     0.8828    0.8763    0.8744       671
          weighted avg     0.8843    0.8703    0.8722       671

```