# C4功率形状正式机制冻结：五方法十五模型

P患者条件表征12模型已全部完成且主概率/直接Fraction/辅助日志/权重审计通过；全部8方案未达严格目标，未确认跨患者、标准SupCon或患者内目标提供原Mom以上的可靠增量。保留原M方向矩及原CE食谱，不将未确认aux带入新轮。其阴性源码/权重/逐图概率完整保留，不恢复旧训练。

机制：µ/σ丢失了四方向响应的周期分配。对原stem四响应r，a=(r0−r2)/2，b=(r1−r3)/2，c=(r0−r1+r2−r3)/4，E1=(a²+b²)/2，E2=c²，人口var=E1+E2。q=(E2−E1)/(E1+E2+1e−6)提供µ/σ不足以确定的信息，并在四循环方向变换下不变；强度缩放因eps只有近似不变，低能量有限。假设这种方向功率结构能补足CLE纹理/形态表征，只是可检验假设，不能把频率量直接诊断为组织形态。

五arm×配对seed45/46/47全部独立公开初始化60epoch：mom_shape=原Mom+q零投影；shape_only去掉完整18528参数Mom适配器，仅原base+q；mom_permuted同参数将响应固定排列[0,1,3,2]破坏方向邻接；mom_duplicate同额外分支但重复原σ，控制重参数化；mom_rawh2=原Mom+成熟DFT第二频率幅值sqrt(max(E2,1e−12))，控制功率对比/归一化相对原幅值是否必要。对原Mom是不加q的去一/增量比较；完整去Mom、错方向、重复信息、成熟幅值均明确注册，非无限枚举。

额外96→96零初始化1×1投影9312参数，复用四滤波响应，保留原base权重对象、µ/σ计算、原Mom分支。所有构造fork RNG，原公开起点logit/参数/配对seed严格一致，无旧训练权重热启动。原Mom保留分支的四arm27850275参数/9.169643520G；shape_only预计27831747/9.054038016G，必须CPU/CUDA实测确认，不隐瞒去一预算不同。逻辑FLOPs排除功率/除法/归一化/访存，实际吞吐必报。

固定B0和A–C清单不变。原train2077/182、val427/39、test576/41，每epoch2077唯一训练索引一次，物理3595cache不等于训练用3595。原CE、batch32、encoder1e−5、新模块/头1e−4、AdamW.05、cosine0、clip1、AMP、HV/rotation15、worker2/thread1；不改标签/等级阈值/split/筛图/重采样。每arm先实测单/并发GPU/显存/CPU/加载/I/O/RAM/吞吐，97%最佳吞吐内最小并发、6GiB余量，GPU满载时不制造无益任务。

原0度val最早最高macroF1选checkpoint；每完整三seed统一test单视图及C4概率，两个独立加载路径全427/576概率、有限权重、source/task/checkpointSHA、全部60epoch索引和患者隔离独立审计。报告macroF1/accuracy/恶性召回/患者等权accuracy/混淆矩阵/成本，全部seed和阴性保留。直接比较mom_shape对原Mom/shape_only/mom_permuted/mom_duplicate/mom_rawh2/B0，single/C4分列，20k共享41患者配对bootstrap与独立整数Fraction均值严格>3/200、所有seed正、区间下界正，不挑最好seed。

CPU原型已通过NumPy FFT/Parseval、同µσ不同q反例、C4与输入90°局部空间协变/错序破坏、退化/低能量有限、eps缩放偏差、doublegradcheck、官方H-Net纯幅值输出/输入梯度和公开Tiny224/参数ID/RNG/新投影有限反向。正式CPU还须新增shape_only真实去掉Mom对象/参数的检查，再真实CUDA batch32两次有限/非零投影梯度、六B0/Mom完整427概率复建、cacheSHA和训练加载拒绝test，之后才正式训练。数值证明不替代CLE效果。

H-Net/群池化/DFT幅值均已有先验，原型保留MIT官方blob与源码。这里是离散C4后处理，不是完整H-Net/连续旋转网络或ECCV2018整法复现；普通尾部不约束，不能称整网旋转等变/不变。共享FMA是空间rfft2，与方向轴变换不同，不将它包装为本算子。原始启动器哈希类型错误及独立修复已保留，未修改原型冻结算子/配置，证明实际在修复后通过。

用户授权test反馈下一设计，历史test已查看、AC构造用标签，所有结果为自适应内部开发；bootstrap仅条件描述，不校正选择偏差，不称独立泛化/新外部验证。等级仅一致性代理。不保证提点、SCI创新或录用。一般阴性据对照继续整体研究。
