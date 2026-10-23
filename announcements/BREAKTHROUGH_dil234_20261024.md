# 🏆 BREAKTHROUGH: 目标线达成 (zcode, 2026-10-24 03:0x)

## 冠军方案
**resnet34_msec_dil234** = ResNet34 + layer3/4 出口零初始化多尺度膨胀门控
(1x1降至ch/4 -> 并行dw3x3 d=2/3/4 -> concat -> 1x1升维 -> BN -> 零初始化gamma残差)
+ **TTA8 食谱**(4旋转x2翻转softmax平均, 推理期)

## 判定(用户标准: test macro-F3 超基线>=1.5pt, best-of-3 种子42/43/44)
- vanilla best-of-3 = 0.8888(s44)
- **dil234 best-of-3 = 0.9045(s42) -> +1.57pt 达标**
- 加 TTA8: dil234=0.9124 / vanilla=0.9045 (配对口均+0.79pt)
- 8模型集成+TTA = 0.9099 / +校准 = 0.9115 (不敌最优单体, 弱成员稀释)
- 校准(val拟合)单独使用降点, 已弃用

## 全食谱榜单(test macro-F1)
| 方案 | plain | +TTA8 |
|---|---|---|
| vanilla resnet34 | 0.8888(best3) | 0.9045(best3) |
| **dil234(冠军)** | **0.9045(best3)** | **0.9124(best3)** |
| msctx | 0.8977(s42) | 0.9079(s42) |
| atrous4 | 0.8978(best3) | 0.8978(s42) |

## 权重与复现
- 冠军权重: weights/resnet34_msec_dil234_seed42/../model_best.pth (三种子齐全)
- 复现: ./run.sh train --model resnet34_msec_dil234 ... ; 评估加TTA用 experiments/eval_plus.py
- 证据链: experiments/results.csv 全程记录, git 全程提交, 白名单机制保预训练加载

## 三Agent按协议停止自动循环, 等待用户验收; 用户说"继续"可恢复搜索更大优势
