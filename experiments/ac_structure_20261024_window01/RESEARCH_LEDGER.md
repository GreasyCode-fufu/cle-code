# CLE 新窗口独立研究账本

更新：2026-10-24T18:15:26.187344+08:00。任务进行中，尚未最终test核验，尚未宣称可靠达标。

## 本次目标与新增授权
固定A–C上相对预冻结竞争baseline，至少3配对seed，最终test平均macroF1严格提升>.015，跨seed稳定且患者配对bootstrap区间下界>0。研究不保证达标、论文创新或录用，不设时间上限，不因几轮benchmark或一般阴性自行结束。用户追加：可以先探索高验证分数DINO；若难有稳定提升，换到其它竞争性基础主干开展公平调优和冻结后的设计。所有baseline阶段转换公开记录，最终test前固定最终方案。

## 环境与历史证据
SSH ubuntu-server实际Dell-T2，uid1000 login；指定项目/models/shared module_library/datasets均login:login，原Git无他人未提交变更。远端适用路径未发现额外AGENTS.md，遵循本聊天提供的AGENTS。RTX4090总49140MiB，24逻辑CPU，125GiB RAM。未干预其它Agent。已读完整历史RESEARCH_LEDGER与最终AC报告，旧停止状态仅上一阶段；未恢复旧完整数据研究。共享库只读，旧源码/配置/checkpoint不改。

## 数据和访问审计
冻结train2077/182患者、val427/39、test576/41，清单SHA b136e3a48ff1b6442957797b4d98228dfc7cedc9abae4f456052dcfcf4f69de8。患者交集为空，不重新分级、不改阈值/标签/清单。.cache_data_study物理3595图，完整缓存SHA已核验；新GuardedDataset限制实际访问，训练仅train2077、选择仅val427，每epoch校验实际训练索引和次数；预检确认test索引会被拒绝。本轮没有加载test图像或新test预测。

历史test已查看，且AC定义使用其标签；只能描述固定选择后AC分布，不能称全新外部验证。等级是模型一致性代理，不能证明医生标签全正确。val已反复开发，bootstrap条件于选择后的模型，不能涵盖全部筛选/训练不确定性。

## baseline与首轮机制冻结
依据开发均值选择DINO full（0.982784），CNN原参照.973445，未依据历史test排序。3个DINO checkpoint按任务签名、源码/config SHA、权重SHA和60日志/最早最佳轮核验复用，全部427概率独立重现，避免重复有效训练。第一轮同60epoch/batch32/encoder1e-5/head及readout1e-4/AdamWwd.05/cosine/clip1/FP16/原增强。

自主relational在第4/8/12层patch上以共享96维投影学习局部邻接乘积统计，零起点残差加到CLS+patch均值；所有主干继续微调。重训no_neighbor/no_depth/final_only三个等参数对照，official4忠实读出公式参照，ConvNeXtBase成熟层级CNN参照，共6×3=18新训练。没有包装成熟组合为已证明新颖性。现代CNN参数/计算额外，非等预算，后续基础baseline选用需公平调优。源码/协议/config/jobs冻结0c6c084，数值预检859e056，资源实测56c5570。

## 数值、库存与资源
静态梳理models 74文件、共享module_library 425文件，均AST可解析。保留完整SHA/类/functions/imports。已核对共享CBAM空间BN/bias与官方差异、SE基本方程、Mona未核准项、DINO官方四层公式与当前timm插值/食谱差异；未核对的文件名会议标签不当真实出版证据。

真实图像两步AMP反向有限、主干和新增投影非零梯度、零初始化与原baseline逐位一致、四读出paired初始化一致、边界邻域NumPy参考、等参数和全部三baseline全val概率复现通过。relational21816486参数，比baseline增加185379；计数FLOPs12.304G对12.247G，额外约.465%，不含norm/池化/逐点乘等。ConvNeXt87569539参数、30.707G。

1/2/4并发12秒共同窗口总吞吐分别1342.7/1341.7/1340.2张/s；更多并发未提高总吞吐，初选最小近最优并发1，正式阶段按GPU空档增至当前3。不是旧6/8硬上限。每2秒GPU计算/显存/CPU总/单核/I/O等待/RAM日志，逐epoch loaderwait/compute/samples/s/耗时。最新GPU100.0%，CPU18.3%，I/O等待0.000%；只代表实际窗口，不把准备/审计间隔称持续满载。

## 当前执行与审计状态
本窗口bootstrap PID790617，当前状态training，已完成4/18；实际pid以status和签名核验。完成模型先保留全部原始val和权重，整轮独立审计尚待全部完成，不能用初步点估计宣布成功。
独立audit_window01.py将严格重加载全部18+3模型、重推每图、复核最佳轮/访问集合/有限参数/源码权重SHA/指标/5比较患者bootstrap/预算/login归属；保存off诊断和阴性。单图原始val CUDA预测已与保存概率逐位相同，CPU烟雾测试通过，未用test。

## 持久工作与交付
专属heartbeat automation id cle，每15分钟继续当前聊天，正常运行与一般阴性安静，真实故障/可靠达标/必要信息通知。会在首轮结束后分析开发错误证据、审计，再实际启动下一预注册机制或其它主干基础baseline阶段。旧报告的“停止”不能终止本目标。不得反复看test或根据test失败改baseline/清单/候选。
完整源、协议、配置、日志、概率、审计、阴性及权重均服务器保存。当前中文报告/结构比较协议/复现使用说明与签名证据同步新窗口outputs；最终报告将随持续研究更新，不把启动报告说成最终达标报告。

最近提交：
```text
241fd41 results: retain new CLE window completed jobs window01_no_neighbor_s45
0bf2255 results: retain new CLE window completed jobs window01_relational_s47
7f9e0ce results: retain new CLE window completed jobs window01_relational_s46
04ce4f5 results: retain new CLE window completed jobs window01_relational_s45
56c5570 perf: measure single and concurrent CLE training throughput before scheduling
859e056 test: pass real-image numerical, loader-boundary and baseline reuse preflight
0c6c084 research: preregister new CLE window baseline, relational depth mechanism and paired controls
be2baed chore: close delivered AC-only dataset research and final model evaluation
```
