# 未进入最终方案的路线

## C2f_SPP

仅部分通道执行标准空间卷积，旁路通道缺少空间更新；全模型 Params/GFLOPs 降幅有限；test mAP50-95≈0.420，被 SR1-P4Only 支配。

## C2f_DSM

主动分支使用标准卷积，廉价分支使用 depthwise 卷积；高 IoU 定位有所恢复，但 test mAP50-95≈0.422，仍被 SR1-P4Only 支配。

## MTRD

基于 Markov 多步转移关系，相对 NoKD 有效，test mAP50-95=0.432186，对 moisture 较有利；整体仍低于 Gram RelKD=0.435680。保留为关系蒸馏消融，不是失败实现，也不是最终主方法。

## GAN / E04

E04 训练未出现 NaN/Inf、梯度爆炸、显存泄漏或 checkpoint 污染，但判别器长期呈 real accuracy≈1、fake accuracy≈0 的单边状态；test mAP50-95=0.420200，低于 Paired NoKD。GAN 不进入最终方法。

## 解释约束

以上代码和历史 checkpoint 被完整保留是为了复核，不代表它们参与 Gram RelKD best.pt 的推理。不得在新对话中默认重新采用 MTRD、GAN、C2f_SPP 或 C2f_DSM 为主方案。

