# 本窗口持续研究接续

目标和用户原始约束继续有效；最新补充见同目录CUMULATIVE_RESEARCH_POLICY_ZH.md：有效机制保留，在其上再研究1–2个互补机制，做累计和去除重训消融。不能把模块数量凑成“已证明SCI创新”。每次迭代用val，最终冻结后可统一test全部预定消融；不循环test调参。

主机ubuntu-server，项目/home/login/data/cle/auto/GPT，Python/home/login/miniconda3/envs/sunfuxiang/bin/python，Git/home/login/miniconda3/bin/git。只操作本窗口新目录，不干预其它Agent。共享module_library只读，不改旧冻结源码/配置/权重，保持login归属。SSH别名以本机当前配置为准。

1. experiments/ac_structure_20261024_window01：第一阶段18新训练+3复用DINO，全部独立审计passed；主候选开发均值+0.159pt但一个seed负、患者区间跨零，未确认结构收益。所有21模型和阴性保留，结果已交付；旧DINO强参照不是最终独立test优胜结论。不要重启该已完成队列。
2. experiments/ac_qualification_20261024_window01：四成熟家族×两食谱×三个seed公平资格比较，24有效/21新+3Base224复用。bootstrap最初797303、queue800906，后续必须用实际/proc/<pid>/cmdline核验，状态文件或旧PID不能当进程证明。原型准备期间GPU训练正常。先检查status、bootstrap_status、失败日志尾、实际进程。不要重复启动或覆盖部分日志。完整训练已complete时只能补推理。
3. 当前目录ac_field_prototype_window01：下一个光学视野支持机制的CPU原型，尚无CLE性能证据。v1在无bias+分数支持时不等同原作者，失败源码/配置/日志和failed_reference_diagnosis.json保留。v2修正为NVIDIA实际无bias公式，numeric_preflight_v2.json已通过20项算子检查+160配对增强；model_integration_cpu.json已通过四家族七接口共28项集成，source/head张量、零起点logits、梯度/SE保留及预算均检查。paired_dataset_audit.json已通过两个食谱全部5008次train/val RGB、标签/权重/索引和读取后RNG完全相同，并通过160个RGBA整体增强的空间mask对齐参照。这些是实现证据，不是分类性能结果。

资格比较完成后，先确认其真实进程退出、independent_audit passed，读取selected_baseline.json。它按预注册质量门槛与最高开发均值选择，不能故意挑弱模型。将选定基础、完整食谱、三个checkpoint和比较规则在正式结构阶段锁定；DINO/所有资格结果保留。若无合格者，继续公平调优，不能伪选弱baseline或结束整体目标。

下一结构阶段可在选定主干上验证本支持机制：原baseline、field_pool、原方程reference、fov_residual、no_field、no_local_norm、no_pool。复制已核验原型到新的独立正式源码目录，保留原型版本；将输入RGB切片显式contiguous以匹配原RGB加载器内存布局，再做真batch32 CUDA/AMP、全部427 baseline概率复现、同seed输入/RNG、训练索引、参数和显式功能算子FLOPs/延迟/显存预检并提交后训练。原型CPU不能替代这些门槛。新增gate LR需先按训练数值规模预注册并给相应控制同预算；原encoder/head食谱不暗改。不能从已经拟合的B1再多训而让baseline只训原预算。

等待资格比较时可继续实现审计/训练/统计的独立源码与数值检查，但不能提前选baseline或把未训练的原型当有效第一组件。第一项机制有有效证据后登记固定，再依据剩余错误加入第二/可选第三项；保留工作核心和累计/drop-one重训消融。一次阴性不停止总体研究，也不无限枚举近似组合。

唯一AC train2077/182、val427/39、test576/41；不重分级、筛图、改标签/阈值。原缓存3595行，实际训练推理只许冻结train/val索引；新448缓存2504行只含AC train+val且raw SHA核验。完整文件SHA会遍历缓存字节，不等于其所有图都用于训练。历史test被使用、AC定义用过test标签，不能叫全新外部验证。若用患者OOF需明确它仍条件于AC选择。

成功须至少三配对seed的固定test宏F1平均增益严格>.015、跨seed稳定且患者配对区间下界>0；同时报告accuracy、恶性召回、患者等权准确率、混淆矩阵与成本。最终方案/所有对照/权重SHA/源/规则先冻结，再统一核验test，不按结果返回调参。未满足不能宣布达标或保证创新/录用。

持续每2秒GPU/显存/CPU总和单核/I/O等待/RAM，逐epoch loader等待/compute/样本每秒/耗时；resource_observer.py已给资格阶段补充物理磁盘带宽/繁忙比例和本流水线进程I/O。不为CPU数字增加无用训练，GPU满载不挤占；并发依据实测与显存余量而非旧6/8上限。

Git精确暂存本窗口路径。资格调度器会在任务完成时提交；其它目录的手动提交需选择所有仍活任务距离结束有余量、索引为空的窗口，避免暂存不同路径被严格调度器误认。保留有效结果、源码和权重SHA。中文报告/结构/复现/使用说明及失败版本同步C:/Users/a/Documents/Codex/2026-10-24/cle-cle-a-c-baseline-f1/outputs。正常运行与一般阴性不做例行通知；仅可靠达标、真实未解决故障或必要用户信息通知。目标继续active，不因阶段完成擅自complete/paused/blocked。
