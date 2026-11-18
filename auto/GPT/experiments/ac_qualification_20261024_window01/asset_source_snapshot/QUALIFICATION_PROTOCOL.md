# 新窗口第二阶段：公平选择其它竞争性基础主干

身份：ac_qualification_20261024_window01。这是基础模型与食谱资格比较，不是自主模块贡献实验。用户追加允许DINO难提点时转向其它基础baseline。第一阶段DINO及全部消融结果保留，不能事后换参照改写第一轮成败。

## 触发和边界

先等待ac_structure_20261024_window01的18项及独立审计结束，确认原bootstrap/queue真实进程已退出、审计passed，主relational相对DINO未通过原strong_dev_signal，才正式训练本阶段。若通过则暂停此备用训练队列，继续第一阶段的有效证据。资产下载、仅train/val缓存及数值源码准备可以先做。没有读取新test图/概率/错误的步骤。触发基于开发证据，不基于历史test排序。

冻结AC清单、原标签/分级/患者split完全不改。训练2077/182患者，val427/39，test576/41仅保留清单元数据并隔离。旧3595物理缓存仍仅通过允许索引读。新高分辨率缓存只含2504张AC train+val，从冻结原图SHA逐一校验后生成448×448 RGB；不混入test或D/E。训练图通道在2504张缓存中全部相同，灰度；黑像素比例中位约23%，仅作为光学视野/细节研究线索，不是患者或样本筛选、噪声标签结论。

## 成熟主干与公平预算

候选四家族：torchvision ConvNeXt-Tiny、EfficientNetV2-S、Swin-T、ConvNeXt-Base，全部官方ImageNet1K权重严格加载1000类模型后替换3类线性头。各家族stochastic depth .1，EfficientNet原dropout .2保留，Swin/ConvNeXt按官方架构。官方参数/权重URL/完整SHA/实际torchvision源码SHA在assets和config中冻结。共享模块不执行、不修改；不把这些成熟模型当新结构。DINO保持原强参照，原CNN均值.973445保持竞争性锚点。

每家族同样两种分辨率食谱×三配对seed45/46/47，共24个有效模型。第一轮ConvNeXt-Base/224三seed按原签名、源码/config/权重SHA/日志/val概率复用，故新增21个完整训练，禁止重做这三项有效训练。四家族相同60epoch、batch32、AdamWwd.05、encoder1e-5/head1e-4、cosine到0、clip1、动态FP16 AMP、CE无平滑和类别/患者权重。未给予候选更多调参预算。两食谱A=原resize256/crop224，B=resize448/crop384；H/V flip、rotate15、随机裁剪和验证中心裁剪完全同机制。全部使用各权重元数据确认相同的ImageNet mean/std。B让EfficientNetV2-S获得官方默认384输入，也让其它家族获得同样高分辨率机会；不把A/B差归因单一新结构。这里是有限、相同预算的资格比较，不声称每个架构已穷尽调优。

本阶段不比较新增模块，不用最高seed。先逐模型取最早的最高val macroF1，再取同pipeline三seed均值。竞争资格：三seed均值至少.973445（原成熟CNN开发锚点），最差seed至少.96。资格内选择均值最高的pipeline；若均值差小于1e-4，依次按更低计数FLOPs、参数、模型名称确定，不通过故意挑弱模型获得后续空间。若全部不合格，不冻结弱baseline，依据训练/错误证据安排另一轮同等调优或其它主干。全部24模型及DINO/CNN参照公开，不能只给获胜模型。

选出的基础baseline、完整食谱、3checkpointSHA及指标规则在结构优化前写不可变selected_baseline.json并Git提交。下一结构阶段在这个已冻结基础上提出明确不同机制与重训消融；baseline继续获得与候选同等训练/调参机会。不能把资格筛选本身包装成网络创新。最终test前再统一冻结主要baseline和最终候选，之后不因test换baseline或清单。

## 指标、审计与资源

macroF1为主，accuracy、恶性召回、患者等权accuracy、混淆矩阵、参数、计数FLOPs、延迟/样本量每秒/时间/显存全部报告。20,000次患者配对bootstrap只描述开发比较不确定性，资格筛选/反复val与AC筛选偏差不能由区间消除。先配置/源/jobs签名、真实图像有限AMP/优化器覆盖/paired跨分辨率初态/旧Base224全部val重现/缓存SHA及访问隔离检查并提交，再实测并发并开训练。Swin的F.linear与attention成本显式计入，不能漏报。

调度用实测单任务与并发吞吐、每任务显存和CPU上限推导；不沿用旧6/8硬上限。GPU已满不堆无用任务。每2秒资源与每epoch loader等待、compute、samples/s、训练耗时；只调度本窗口有效任务。故障查日志及实际同流水线进程，完成训练只补推理；不覆盖部分日志盲重训。独立审计严格加载、重推全部val、源码与权重SHA、最早最佳轮、有限参数、固定清单/患者互斥、指标、预算/所有权，保存所有阴性及资格失败。研究仍在继续，不因本阶段结束自行宣布目标完成。

官方源核对（2026-10-24浏览并与安装版元数据核对）：
- https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
- https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
- https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py

与本地库差异：本地Convnext.py是Meta格式实现，不能直接装入torchvision格式权重；本轮使用torchvision原构造和strict状态。共享Partial Convolution类文件的名称不能直接等同于mask-normalized inpainting PartialConv，未来若研究光学视野需另查原公式和边界，不能依据模块名混用。本阶段不引入该分支。
