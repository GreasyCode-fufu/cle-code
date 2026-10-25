# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/resnet34_seed43/resnet34/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9431 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9925 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9428 |
| Balanced Acc | 0.9415 |
| Macro-Sensitivity | 0.9415 |
| Macro-Specificity | 0.9707 |
| Macro-Precision | 0.9457 |
| AUC-PR (Macro) | 0.9869 |
| Cohen's Kappa | 0.9137 |
| MCC | 0.9142 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 1652.9 (0.60 ms/张) |
| End-to-End FPS | 248.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9402 | 0.8995 | 0.9227 | 0.9692 | 0.9875 | 0.9805 | 208 |
| Malignant_lesions | 0.9034 | 0.9850 | 0.9636 | 0.9326 | 0.9586 | 0.9915 | 0.9830 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9870 | 0.9740 | 0.9740 | 0.9870 | 0.9986 | 0.9971 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=22 TN=346 FN=11
- **Malignant_lesions**: TP=159 FP=6 TN=394 FN=17
- **body_of_stomach_normal**: TP=187 FP=5 TN=379 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 6 | 5 |
| Malignant_lesions | 17 | 159 | 0 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8995    0.9471    0.9227       208
     Malignant_lesions     0.9636    0.9034    0.9326       176
body_of_stomach_normal     0.9740    0.9740    0.9740       192

              accuracy                         0.9427       576
             macro avg     0.9457    0.9415    0.9431       576
          weighted avg     0.9439    0.9427    0.9428       576

```