# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce4m_seed42/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7955 |
| Accuracy | 0.8406 |
| Macro-AUROC (OvR) | 0.9614 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8364 |
| Balanced Acc | 0.7978 |
| Macro-Sensitivity | 0.7978 |
| Macro-Specificity | 0.9669 |
| Macro-Precision | 0.7986 |
| AUC-PR (Macro) | 0.8642 |
| Cohen's Kappa | 0.7946 |
| MCC | 0.7958 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9189.1 (0.11 ms/张) |
| End-to-End FPS | 560.1 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8241 | 0.9943 | 0.7354 | 0.7773 | 0.9966 | 0.9928 | 0.8849 | 199 |
| cancer | 0.9267 | 0.9496 | 0.8588 | 0.8914 | 0.9751 | 0.9882 | 0.9664 | 2645 |
| empty | 0.9963 | 0.9603 | 0.9172 | 0.9551 | 0.9983 | 0.9880 | 0.9615 | 3256 |
| normal | 0.7256 | 0.9816 | 0.8848 | 0.7973 | 0.9483 | 0.9508 | 0.8733 | 1735 |
| other | 0.6290 | 0.9528 | 0.7195 | 0.6712 | 0.9303 | 0.8901 | 0.7431 | 1717 |
| stroma | 0.6848 | 0.9628 | 0.6761 | 0.6804 | 0.9642 | 0.9583 | 0.7562 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=164 FP=59 TN=10376 FN=35
- **cancer**: TP=2451 FP=403 TN=7586 FN=194
- **empty**: TP=3244 FP=293 TN=7085 FN=12
- **normal**: TP=1259 FP=164 TN=8735 FN=476
- **other**: TP=1080 FP=421 TN=8496 FN=637
- **stroma**: TP=741 FP=355 TN=9197 FN=341

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 164 | 3 | 2 | 8 | 18 | 4 |
| cancer | 8 | 2451 | 1 | 45 | 86 | 54 |
| empty | 0 | 0 | 3244 | 0 | 12 | 0 |
| normal | 4 | 175 | 3 | 1259 | 146 | 148 |
| other | 42 | 140 | 287 | 19 | 1080 | 149 |
| stroma | 5 | 85 | 0 | 92 | 159 | 741 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7354    0.8241    0.7773       199
      cancer     0.8588    0.9267    0.8914      2645
       empty     0.9172    0.9963    0.9551      3256
      normal     0.8848    0.7256    0.7973      1735
       other     0.7195    0.6290    0.6712      1717
      stroma     0.6761    0.6848    0.6804      1082

    accuracy                         0.8406     10634
   macro avg     0.7986    0.7978    0.7955     10634
weighted avg     0.8375    0.8406    0.8364     10634

```