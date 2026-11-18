# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/pvt_tiny_seed44/pvt_tiny/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9708 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9969 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9764 |
| Balanced Acc | 0.9653 |
| Macro-Sensitivity | 0.9653 |
| Macro-Specificity | 0.9868 |
| Macro-Precision | 0.9771 |
| AUC-PR (Macro) | 0.9923 |
| Cohen's Kappa | 0.9626 |
| MCC | 0.9628 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 641.0 (1.56 ms/张) |
| End-to-End FPS | 201.7 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9698 | 0.9648 | 0.9746 | 0.9868 | 0.9963 | 0.9954 | 195 |
| Malignant_lesions | 0.9114 | 0.9943 | 0.9730 | 0.9412 | 0.9802 | 0.9944 | 0.9817 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=7 TN=225 FN=3
- **Malignant_lesions**: TP=72 FP=2 TN=346 FN=7
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 2 | 1 |
| Malignant_lesions | 7 | 72 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9648    0.9846    0.9746       195
     Malignant_lesions     0.9730    0.9114    0.9412        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9766       427
             macro avg     0.9771    0.9653    0.9708       427
          weighted avg     0.9766    0.9766    0.9764       427

```