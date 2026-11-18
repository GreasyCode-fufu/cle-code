# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_small_seed44/convnext_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9801 |
| Accuracy | 0.9836 |
| Macro-AUROC (OvR) | 0.9966 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9835 |
| Balanced Acc | 0.9780 |
| Macro-Sensitivity | 0.9780 |
| Macro-Specificity | 0.9911 |
| Macro-Precision | 0.9824 |
| AUC-PR (Macro) | 0.9862 |
| Cohen's Kappa | 0.9739 |
| MCC | 0.9739 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.41M |
| Pure-Inference FPS | 411.0 (2.43 ms/张) |
| End-to-End FPS | 170.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9828 | 0.9796 | 0.9821 | 0.9870 | 0.9944 | 0.9959 | 195 |
| Malignant_lesions | 0.9494 | 0.9943 | 0.9740 | 0.9615 | 0.9886 | 0.9954 | 0.9628 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=4 TN=228 FN=3
- **Malignant_lesions**: TP=75 FP=2 TN=346 FN=4
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 2 | 1 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9796    0.9846    0.9821       195
     Malignant_lesions     0.9740    0.9494    0.9615        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9836       427
             macro avg     0.9824    0.9780    0.9801       427
          weighted avg     0.9835    0.9836    0.9835       427

```