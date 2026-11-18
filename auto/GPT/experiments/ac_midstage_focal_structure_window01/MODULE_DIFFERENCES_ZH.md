# 中间层算子来源与接口

midstage_focal.py与c4_moment_stem.py逐字复制已通过无CLE图像CPU原型。与SAL末层读出不同，本轮在第二stage192×28×28加零特征残差，由原后续网络处理。新增11832参数，五arm同参数但交互函数不同，不等同完全匹配有效表达容量。up首步正，down/core首步0第二步正，原Mom首步另检。计数含新全部Conv/DW与Mom三额外方向卷积，排除LN/GELU/中心化/逐元素/replicate/访存。

官方FocalNet MIT原字节与受限forward算子oracle在CPU原型references保留；未执行官方整网/训练入口。rank压缩、biasless f/h、replicate边界、中心化、LN、中间零残差是适配，成熟Focal/Mom组合不自动创新。不承诺整CNN平移或旋转等变/临床定位/稳健。原Mom/B0和旧SAL/COV有效全轮不重训，旧冻结文件/共享库只读。最终2–3网络设计组合累计+1.5，不设单点+1.5门槛。所有test自适应内部开发，条件CI不校正选择偏差。
