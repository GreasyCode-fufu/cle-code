# Gram RelKD 模型卡

## 正式 test split 指标

| 指标 | 数值 |
|---|---:|
| Precision | 0.749849 |
| Recall | 0.675435 |
| mAP50 | 0.704466 |
| mAP75 | 0.459626 |
| mAP50-95 | 0.435680 |
| spalling AP50-95 | 0.582001 |
| crack AP50-95 | 0.410681 |
| moisture AP50-95 | 0.314357 |

## 模型复杂度

- fused Params：2,603,129
- unfused Params：2,607,913
- GFLOPs（640）：7.044096
- best.pt：5,480,483 bytes，约 5.227 MiB

## 严格配对 NoKD 对照

NoKD：P=0.76923、R=0.66821、mAP50=0.69747、mAP75=0.43523、mAP50-95=0.42120；spalling/crack/moisture AP50-95=0.56010/0.41213/0.29136。

Gram RelKD 相对 NoKD：mAP50 +0.00700、mAP75 +0.02440、mAP50-95 +0.01448（1.448 个百分点）、spalling +0.02190、crack -0.00144（基本持平）、moisture +0.02299。

配对证据：Student 初始参数 SHA256 均为 `b5c987921cb2eae9d9536c8c632c619145e7d56a992f37d9badff8a368ae25b6`；seed、数据配置、Student YAML、调用链与全部训练超参数一致，除输出名称外唯一优化变量为 `kd_weight`（0 vs 0.0431658384）。

## 适用范围和限制

该 checkpoint 面向 Hong Kong 墙体缺陷三类检测。正式证据仅覆盖单 seed；crack 没有获得 AP50-95 增益；标签宽松与类别不平衡可能影响解释。部署时不需要 `ultralytics_ext`，但复现实验仍依赖外部数据集路径和现有 yolo conda 环境。

> FINAL MAIN METHOD = GRAM RELKD

