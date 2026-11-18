# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/vit_base_patch16_224_seed42/vit_base_patch16_224/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8482 |
| Accuracy | 0.8743 |
| Macro-AUROC (OvR) | 0.9779 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8779 |
| Balanced Acc | 0.8638 |
| Macro-Sensitivity | 0.8638 |
| Macro-Specificity | 0.9754 |
| Macro-Precision | 0.8455 |
| AUC-PR (Macro) | 0.9129 |
| Cohen's Kappa | 0.8421 |
| MCC | 0.8450 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 275.6 (3.63 ms/张) |
| End-to-End FPS | 195.4 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8700 | 0.9948 | 0.7500 | 0.8056 | 0.9977 | 0.9974 | 0.9059 | 100 |
| cancer | 0.7660 | 0.9856 | 0.9417 | 0.8448 | 0.9327 | 0.9617 | 0.9313 | 1329 |
| empty | 0.9963 | 0.9961 | 0.9902 | 0.9932 | 0.9985 | 0.9999 | 0.9998 | 1621 |
| normal | 0.9369 | 0.9913 | 0.9586 | 0.9476 | 0.9864 | 0.9955 | 0.9837 | 1014 |
| other | 0.7531 | 0.9722 | 0.8317 | 0.7904 | 0.9558 | 0.9585 | 0.8617 | 879 |
| stroma | 0.8603 | 0.9122 | 0.6007 | 0.7075 | 0.9770 | 0.9547 | 0.7950 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=87 FP=29 TN=5573 FN=13
- **cancer**: TP=1018 FP=63 TN=4310 FN=311
- **empty**: TP=1615 FP=16 TN=4065 FN=6
- **normal**: TP=950 FP=41 TN=4647 FN=64
- **other**: TP=662 FP=134 TN=4689 FN=217
- **stroma**: TP=653 FP=434 TN=4509 FN=106

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 87 | 1 | 2 | 2 | 4 | 4 |
| cancer | 19 | 1018 | 1 | 17 | 43 | 231 |
| empty | 1 | 0 | 1615 | 0 | 4 | 1 |
| normal | 4 | 22 | 0 | 950 | 9 | 29 |
| other | 3 | 21 | 13 | 11 | 662 | 169 |
| stroma | 2 | 19 | 0 | 11 | 74 | 653 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7500    0.8700    0.8056       100
      cancer     0.9417    0.7660    0.8448      1329
       empty     0.9902    0.9963    0.9932      1621
      normal     0.9586    0.9369    0.9476      1014
       other     0.8317    0.7531    0.7904       879
      stroma     0.6007    0.8603    0.7075       759

    accuracy                         0.8743      5702
   macro avg     0.8455    0.8638    0.8482      5702
weighted avg     0.8928    0.8743    0.8779      5702

```