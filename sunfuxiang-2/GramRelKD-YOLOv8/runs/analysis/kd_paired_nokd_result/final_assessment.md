# 最终评估

**Training status: SUCCESS**

SR1-P4Only Paired NoKD 已按要求完成 300 epochs，并在与 RelKD 完全相同的 Hong Kong test split 流程上完成独立评估。严格配对核验通过。

最终证据显示，RelKD 相比 Paired NoKD：

- test mAP50-95：0.42120 → 0.43568，提升 0.01448；
- test mAP75：0.43523 → 0.45963，提升 0.02440；
- Recall：0.66821 → 0.67544，提升 0.00722；
- spalling AP50-95：提升 0.02190；
- moisture AP50-95：提升 0.02299；
- crack AP50-95：下降 0.00144；
- Precision：下降 0.01938；
- fused Params 与 GFLOPs 均不变。

因此，RelKD 在严格配对条件下取得明确的总体收益，尤其改善高 IoU 定位性能和 spalling/moisture 两类表现。结果不是由初始化或训练配置不一致导致。未启动任何其他 KD 权重实验。
