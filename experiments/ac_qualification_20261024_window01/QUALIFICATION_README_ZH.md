# 第二阶段复现与衔接

专属目录：`/home/login/data/cle/auto/GPT/experiments/ac_qualification_20261024_window01`。Python为 `/home/login/miniconda3/envs/sunfuxiang/bin/python`。阶段已由本窗口流水线自动启动，先读status/bootstrap_status并核对实际`/proc/<pid>/cmdline`，不要重复开bootstrap。

四个成熟主干各两种输入食谱、三个配对seed，总24模型；21项新训练、3项ConvNeXtBase224按原签名和权重SHA复用。两食谱均60epoch/batch32，分别resize256/crop224和resize448/crop384；正式训练前配置、任务、原权重/清单/源码SHA、真实图像数值预检已冻结。源实现的训练循环直接复用不可改第一轮engine.worker，在新进程绑定新model/loader/STUDY；没有修改旧源码或其它进程全局变量。

```bash
cd /home/login/data/cle/auto/GPT
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
Q=experiments/ac_qualification_20261024_window01
# 新环境复现入口；现有流水线存活时不要再次运行
$PY -u $Q/bootstrap_qualification.py
# 指定任务幂等入口；完整训练已完成时只补未完成推理，不重训
$PY $Q/run_qualification.py worker --job qualification_swin_t_B_s45
# 所有任务完成后单独复核，不在结构阶段随意重写已锁定审计文件
$PY $Q/audit_qualification.py
```

权重：`weights/ac_qualification_20261024_window01/qualification_<family>_<A或B>_s<seed>/model_best.pth`。使用前必须按complete.json核验SHA。三份复用Base224仍使用第一轮原路径，不复制覆盖。公开预训练和1.5GB高分辨率缓存保留本目录pretrained/cache448，均不入Git，完整SHA及索引在manifest/meta文件。新缓存只包含2504张AC train/val，逐张原图SHA核对，不含test/D/E。

每次实际训练访问2077个AC train索引，val427仅做最早最佳checkpoint选择；每epoch保存索引集合/计数哈希与loader等待、compute、样本每秒、GPU峰值、时间。资源CSV每2秒保存GPU/显存/CPU总和单核/I/O等待/RAM；独立observer补充物理磁盘read/write带宽、繁忙比例和本流水线进程I/O计数。observer不参与模型或随机数生成。

阶段完成后，独立审计会重推24模型全部427张val，生成qualification_results.json、independent_audit.json、selected_baseline.json、QUALIFICATION_REPORT_ZH.md。只在均值>=.973445且最差seed>=.96的候选中按最高均值选择；均值差<.0001时按FLOPs/参数/名称顺序。没有合格者就继续公平调优，不选弱baseline。selected_baseline不可在后续审计中改变。

该选定baseline在下一轮结构比较前再次连同完整食谱、三checkpoint和规则冻结；不会依据新test换baseline。DINO强参照、第一轮阴性和本阶段全部pipeline都保留。下一步仍需适合CLE的明确结构机制、重训消融和独立审计，而非把资格比较当整体目标完成。

尚未提供本阶段默认推理模型，因为基础baseline资格比较仍在进行。最终选定后，会提供绑定指定完整checkpoint、预处理和类别映射的独立使用入口并验证CUDA/CPU；当前第一轮研究模型接口仍可用。

辅助原方法审计partial_method_equation_audit.json已确认：共享21号PConv是FasterNet通道部分卷积，和NVIDIA按有效掩膜重新归一化的PartialConv不是同一算法。共享类与作者类前向/输入及权重梯度逐位一致；NVIDIA真实算子与独立标量公式最大误差为机器精度量级，并通过全空掩膜输出零检查。这不是CLE收益或整套inpainting网络复现的证明。
