# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r5/convnext_v6f_seed44/convnext_v6f/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8783 |
| Accuracy | 0.8735 |
| Macro-AUROC (OvR) | 0.9760 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8725 |
| Balanced Acc | 0.8899 |
| Macro-Sensitivity | 0.8899 |
| Macro-Specificity | 0.9326 |
| Macro-Precision | 0.8714 |
| AUC-PR (Macro) | 0.9653 |
| Cohen's Kappa | 0.8015 |
| MCC | 0.8028 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 846.0 (1.18 ms/张) |
| End-to-End FPS | 273.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8122 | 0.9495 | 0.8889 | 0.8488 | 0.9104 | 0.9648 | 0.9437 | 197 |
| normal | 0.8659 | 0.8927 | 0.8755 | 0.8707 | 0.8844 | 0.9651 | 0.9587 | 276 |
| sever | 0.9917 | 0.9556 | 0.8500 | 0.9154 | 0.9978 | 0.9981 | 0.9934 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=160 FP=20 TN=376 FN=37
- **normal**: TP=239 FP=34 TN=283 FN=37
- **sever**: TP=119 FP=21 TN=452 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 160 | 33 | 4 |
| normal | 20 | 239 | 17 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8889    0.8122    0.8488       197
      normal     0.8755    0.8659    0.8707       276
       sever     0.8500    0.9917    0.9154       120

    accuracy                         0.8735       593
   macro avg     0.8714    0.8899    0.8783       593
weighted avg     0.8748    0.8735    0.8725       593

```