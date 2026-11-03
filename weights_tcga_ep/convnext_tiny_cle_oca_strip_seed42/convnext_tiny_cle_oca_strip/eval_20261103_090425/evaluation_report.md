# convnext_tiny_cle_oca_strip 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_tiny_cle_oca_strip_seed42/convnext_tiny_cle_oca_strip/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8430 |
| Accuracy | 0.8750 |
| Macro-AUROC (OvR) | 0.9700 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8770 |
| Balanced Acc | 0.8513 |
| Macro-Sensitivity | 0.8513 |
| Macro-Specificity | 0.9755 |
| Macro-Precision | 0.8492 |
| AUC-PR (Macro) | 0.8926 |
| Cohen's Kappa | 0.8430 |
| MCC | 0.8465 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.47G |
| FLOPs | 8.94G |
| Params | 27.84M |
| Pure-Inference FPS | 516.4 (1.94 ms/张) |
| End-to-End FPS | 292.1 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7700 | 0.9961 | 0.7778 | 0.7739 | 0.9959 | 0.9926 | 0.8601 | 100 |
| cancer | 0.7690 | 0.9902 | 0.9596 | 0.8538 | 0.9338 | 0.9706 | 0.9385 | 1329 |
| empty | 0.9907 | 0.9993 | 0.9981 | 0.9944 | 0.9963 | 1.0000 | 0.9999 | 1621 |
| normal | 0.9507 | 0.9778 | 0.9026 | 0.9260 | 0.9892 | 0.9931 | 0.9752 | 1014 |
| other | 0.7053 | 0.9733 | 0.8278 | 0.7617 | 0.9477 | 0.8965 | 0.8279 | 879 |
| stroma | 0.9223 | 0.9166 | 0.6295 | 0.7483 | 0.9871 | 0.9671 | 0.7538 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=77 FP=22 TN=5580 FN=23
- **cancer**: TP=1022 FP=43 TN=4330 FN=307
- **empty**: TP=1606 FP=3 TN=4078 FN=15
- **normal**: TP=964 FP=104 TN=4584 FN=50
- **other**: TP=620 FP=129 TN=4694 FN=259
- **stroma**: TP=700 FP=412 TN=4531 FN=59

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 77 | 1 | 0 | 5 | 10 | 7 |
| cancer | 17 | 1022 | 0 | 47 | 78 | 165 |
| empty | 0 | 1 | 1606 | 0 | 12 | 2 |
| normal | 2 | 12 | 0 | 964 | 3 | 33 |
| other | 1 | 23 | 3 | 27 | 620 | 205 |
| stroma | 2 | 6 | 0 | 25 | 26 | 700 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7778    0.7700    0.7739       100
      cancer     0.9596    0.7690    0.8538      1329
       empty     0.9981    0.9907    0.9944      1621
      normal     0.9026    0.9507    0.9260      1014
       other     0.8278    0.7053    0.7617       879
      stroma     0.6295    0.9223    0.7483       759

    accuracy                         0.8750      5702
   macro avg     0.8492    0.8513    0.8430      5702
weighted avg     0.8930    0.8750    0.8770      5702

```