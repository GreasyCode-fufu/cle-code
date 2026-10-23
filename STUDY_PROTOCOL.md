# GPT CLE 自动数据分级与对照实验（冻结协议）

用户已确认：ResNet34；患者级5折 × 种子42/43/44；沿用原A–E阈值；依次比较全量、A–C、等级加权；完整测试集不筛选。原数据、其他Agent目录和GPU进程不修改。

## 数据与独立性

- 数据源：本工作区datasets/cle/{train,val,test}。先逐图SHA256冻结manifest。
- 匿名分组：上游csv/*_processed.csv的leakage_group_id；不保存患者姓名。
- train/val/test三个分组集合必须互斥，缺失或冲突直接停止。
- 分级仅覆盖train。外层使用StratifiedGroupKFold(5)，每个种子每张训练图片恰好得到一次OOF预测。
- 外层剩余患者再用分组10折取一个有三类的内层选模集。内层选模集与训练、外层评估组互斥。外层标签不参与checkpoint选择。
- 原val只用于最终三组训练的checkpoint选择。test在9个最终模型全部训练完后冻结评估，不筛选、不改标、不按test选方法。
- 缓存只做RGB与PIL双线性Resize(256,256)，不使用标签。缓存和原始文件均有SHA256。

## 统一训练

- 本项目原始ResNet34，官方ImageNet resnet34-333f7ec4.pth；只允许分类头缺失，骨干必须完整加载。
- 输入224；Resize256后训练随机水平/垂直翻转、±15度旋转、RandomCrop224；评估CenterCrop224；ImageNet mean/std。
- AdamW lr=1e-4、wd=0.05；60epoch余弦退火至0；batch32；AMP；不早停；按选择集macro-F1保存最佳权重。
- 三种子采用相同初始化、单独DataLoader随机数生成器。默认并行2个自己的任务，各2个DataLoader worker、CPU线程2；不干预其他任务。
- checkpoint目录以job ID隔离，包含SHA256、Git版本、样本清单与训练日志。完成标记最后原子写入，支持断点跳过已验证完成任务；单个任务失败自动重试一次，第二次失败保留日志并报告。

## A–E规则

correct_rate是三个外层OOF预测与现有标签的一致率。confidence是现有标签的OOF概率的均值，不是预测类别最大概率。

判定顺序：A→B→E→D→C。

- A：correct_rate≥0.95且confidence≥0.95。
- B：correct_rate≥0.85且confidence≥0.85。
- E：correct_rate<0.30或confidence<0.30。
- D：correct_rate<0.60或confidence<0.60。
- C：其余。

三次预测下correct_rate仅0、1/3、2/3、1。这是可学性/一致性分级，不是临床确认的标签正确性。

## 最终三组（每组42/43/44各一次）

1. full：全部训练图片、统一权重。
2. ac_only：只保留A/B/C，原标签不变。
3. grade_weighted：保留全部图片，预先固定原始权重A=1、B=1、C=.75、D=.25、E=.10，再在原三分类内分别归一化至权重均值1，避免额外改变原类别总权重；loss=mean(weight×CE)。权重不使用val/test调参。

固定60epoch会让ac_only的总更新步数少于full，这是固定epoch的数据筛选协议，报告中应说明；不根据测试结果补预算。

## 报告与Git

- 原始代码快照、协议代码/manifest、每个OOF完成、分级汇总与下游manifest、每次下游训练完成、冻结test评估和最终报告分别提交Git。
- 原图、缓存、大型权重不进Git；其哈希和路径进入Git。最终报告包含各等级/类别/患者保留情况，三个种子明细、均值与标准差，各类F1、混淆矩阵。
- 不用best-of-3作结论；以同种子的成对差值比较。使用患者分组20000次配对bootstrap，对三个种子差值取均值；注明这是给定已训练模型的条件区间，不覆盖分级/训练集重采样全部不确定性。
- 当前test已在历史研究中被使用过，不能称为新的未触碰外部验证集。
