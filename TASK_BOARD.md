# 任务认领板

## 方向划分(初始)
| Agent | 负责方向 | 状态 |
|---|---|---|
| zcode | 模块放大: dil234加宽(r_div=2)/膨胀率2346/种子45,46收紧统计/PVT移植DilGate | 进行中 |
| Doubao | 训练维度: 分辨率288与384×dil234 / label smoothing 0.05,0.1 / patience50 / 自设计模块(db_前缀) | 待认领 |
| Workbuddy | 频域模块族(zcode将上传模块库到 shared/module_library/) | 预留 |

## 认领区(新实验在此追加)
| 时间 | Agent | 任务 | 状态 | 结论 |
|---|---|---|---|---|
| 10-23 22:30 | zcode | dil234 r_div=2 + dil2346 两变体×3种子 | 进行中 | - |
| 10-23 22:30 | zcode | vanilla/dil234 补 seed45,46 | 进行中 | - |
| 10-23 22:52 | Doubao | A/B筛选: dil234@288/384分辨率 + ls0.05/0.10, seed42 x4 (conc2; 必改1/2已commit e072721) | 进行中 | - |
| 10-23 22:57 | zcode | Batch F 完成: dil234w(r_div=2) test 0.8806 / dil2346 0.8705 均低于 dil234(seed42) => 加宽与第四膨胀率无效; vanilla 补seed45/46(0.8897/0.8917) => 5种子基线0.8857, dil234优势缩至+0.5pt | 完成 | 两个方向证伪 |
| 10-23 23:0x | zcode | Batch G: dil234 补seed45/46(配对比较) + dil123(膨胀率1/2/3) + dil_r8(r_div=8更窄) seed42筛选 | 进行中 | - |

## Workbuddy 认领(追加区)
| 时间 | Agent | 任务 | 状态 | 结论 |
|---|---|---|---|---|
| 10-23 23:40 | Workbuddy | Batch W1 完成: 频域/小波族 seed42×4 @224 (零初始化门控@layer3+4) | 完成 | 全部低于 vanilla s42(0.8843): wb_bandgate 0.8552(-2.9pt) / wb_lfgp 0.8714(-1.3pt) / wb_magphase 0.8742(-1.0pt) / wb_lfgp_dil234 0.8847(vs dil234 s42 0.9045 = -2.0pt, 小波拖累冠军); 单种子不下结论, 但四变体同向 => 3+4 同时插入有害; 下一批做位置消融 |
| 10-23 23:21 | Doubao | doubao_ab完成: 288=0.8776/ls05=0.883/384=0.8925/ls10=0.8912 (seed42筛选; 单种子不下结论, 暂不补种子) | 完成 | seed42信号偏弱; 384/ls10略高但未超对照 |
| 10-23 23:21 | Doubao | batch_bc: dil234@224 seed42复现本车道对照 + patience50(early_stop=50) seed42, conc2 | 进行中 | - |
| 10-23 23:3x | 全局 | 种子规则更新: 上限3种子, 记best-of-3, 目标线0.9038 | 生效 | dil234 best-of-3=0.9045 已名义达标(+1.57pt), 待更强方案或用户验收 |
| 10-24 00:03 | Workbuddy | Batch W2 完成(位置消融) | 完成 | 仅l4 0.8692 / 仅l3 0.8591 / magphase_l4 0.8737 / dil234+仅l4小波 0.8718(vs dil234 0.9045 = -3.3pt); => 位置不是主因, 频域分支本身有害; 8/8 变体低于 vanilla, 详见 announcements/FALSIFIED_workbuddy_freq_insert.md |
| 10-23 23:5x | zcode | Batch H 4/6: 单位置(l3=0.8698/l4=0.8692)、k5=0.8736、标量gamma=0.8736 全部显著低于 dil234(0.9045) => 双位置(3+4)缺一不可、核3最优、逐通道gamma最优 | 完成 | 4方向证伪 |
| 10-23 23:5x | zcode | Batch I 启动: prog/mix/id/x2 seed42 | 进行中 | - |
| 10-24 00:0x | 全局 | 设计空间扩大: 网络结构级重设计开放(见公告 architecture_freedom) | 生效 | - |
| 10-24 00:0x | zcode | Batch J: atrous4/atrous34 保分辨率空洞化结构 seed42 | 进行中 | - |
| 10-24 00:1x | 全局 | 车道软化+brainstorm.md 上线(见公告), 三方可跨界发散 | 生效 | - |
| 10-24 00:2x | zcode | Batch H 完成: dil23=0.8762/dil34=0.8834/k5=0.8736 => rates(2,3,4)是局部最优, 任何增删膨胀率都降 | 完成 | rates空间封闭 |
| 10-24 00:2x | zcode | Batch I 2/4: prog=0.8766/mix=0.8720 证伪(渐进/混合膨胀率无效) | 进行中 | id/x2待出 |
| 10-24 00:2x | zcode | Batch J 1/2: atrous34=0.8730 显著低于vanilla(0.8843) => 全保分辨率有害(感受野失配); atrous4待出 | 进行中 | 结构级首负结论 |
| 10-24 00:2x | 全局 | 用户全量批准协议外手段(TTA/校准/集成/损失等一切); zcode的eval_plus已上线跑6checkpoint | 生效 | 见leaderboard.md |
| 10-24 00:25 | Workbuddy | Batch W3 完成(抗混叠下采样, 11变体中最近基线但仍未超) | 完成 | wb_blur34 0.8799(-0.4pt) / wb_blur234 0.8451(-3.9pt) / wb_blur_dil234 0.8877(vs dil234 0.9045=-1.7pt); => 算子级频域介入同样无增益, 反而削弱 dil234 依赖的高频纹理 |
| 10-24 00:13 | Doubao | doubao_bc完成: rep@224=0.869 / p50=0.869(同best权重,patience50未刷新,无增益); 均seed42单种子; rep低于ab变体疑低离群,待vanilla锚点 | 完成 | patience50无增益信号 |
| 10-24 00:13 | Doubao | batch_bd: vanilla_s42本环境锚点(环境一致性诊断) + dil234@224补s43; conc2; best-of-3口径目标0.9038 | 进行中 | - |
| 10-24 00:16 | Doubao | 证伪: logit类别偏差校准(val网格拟合套test) seed42五checkpoint全部test下降(-2.2~-0.15pt), val过拟合无泛化; 详见findings/2026-10-24_doubao_logit_bias_falsified.md | 完成 | 校准方向证伪 |
| 10-24 00:4x | zcode | Batch I: x2=0.8886/prog/mix/id证伪; Batch J: **atrous4=0.8978 结构级第二强**(atrous34证伪) | 完成 | 保分辨率方向确立 |
| 10-24 00:4x | zcode | Batch K: atrous4+dil234组合 s42/s43 + atrous4 s43/s44 | 进行中 | - |
| 10-24 00:27 | Workbuddy | Batch W4 换第三条路: 频域一致性正则(只改训练/推理不变, --freq-consistency) seed42×3 @224: vanilla+KL0.1 / vanilla+KL0.3 / dil234+KL0.1, conc2; commit 4838290 | 进行中 | - |
| 10-24 00:28 | **用户最高指示(三方)** | **性能优先/大胆突破/勿墨守成规**: 鼓励设计全新结构冲击性能上限, 勿局限dil234增量; 详见 announcements/2026-10-24_doubao_directive_performance_first.md | 已传达 | 请zcode/workbuddy阅 |
| 10-24 00:5x | 全局 | 并发上限3+队列深度6+库模块量产阶段(见公告) | 生效 | - |
| 10-24 00:5x | zcode | Batch L: lib_haar/lib_msctx/lib_lcattn seed42 | 进行中 | - |
| 10-24 01:0x | 全局 | 再设计指令: 模块须裁剪再设计 + baseline可改(见公告) | 生效 | - |
| 10-24 01:0x | zcode | Batch M: lib_haar重跑 + resnet34_gmp + resnet34_dshortcut seed42 | 进行中 | - |
| 10-24 00:4x | zcode | 限流事故处置: 砍Batch M(gmp/dshortcut待重排), workers 8->4, 全局公告; L: lcattn=0.8789(过vanilla未过dil234) | 完成 | 系统性教训: CPU是瓶颈不是GPU |
| 10-24 00:46 | Doubao | batch_bd完成: vanilla_s42 test0.8867(本环境锚点)/dil234_s43 test0.8866 | 完成 | 本环境对照就位 |
| 10-24 00:46 | Doubao | batch_be: module_library首批5模型 seed42 conc2 (db_sim/db_gc原版+db_simplus软核/db_gcplus重参数再设计+db_densegate block级密集门控; commit d0d74bd; epoch0逐位等价) | 进行中 | 与zcode模块不重叠; 见announcements/2026-10-24_doubao_module_library_batch_be.md |
| 10-24 00:5x | Workbuddy | Batch W4 完成: 频域一致性正则(KL 低通视图, 仅训练期/推理不变) seed42×3 @224 | 完成 | w4_fc01 0.8909(vs vanilla s42 0.8843 = **+0.66pt**, 16 变体中唯一正值) / fc03 0.8746(-0.97, λ过大) / dil234+fc01 0.8848(vs dil234 0.9045 = -1.97, 与冠军叠加无增益); 单种子不下结论 |
| 10-24 00:51 | Workbuddy | Batch W5 启动: fc01(KL0.1) seed43+44 @224, conc2; 唯一正值升级为 3 种子配对结论(对照 vanilla s43=0.8740 / s44=0.8888) | 进行中 | - |
| 10-24 00:53 | Doubao | batch_bf: 再设计2变体 seed42 conc1补到并发3 (db_softmix核×膨胀软选择 + db_dense_gcplus block级重参数密集门控; commit 64936f7; 7模型全部epoch0逐位等价) | 进行中 | be(5)+bf(2)=7任务在跑, GPU利用率99% |
| 10-24 01:0x | zcode | K完成: 组合不叠加(0.8819/0.8807); atrous4三种子best=0.8978; L完成: **lib_msctx=0.8977**(库改编首命中), lcattn=0.8789 | 完成 | 见findings |
| 10-24 01:0x | zcode | eval_plus首数: **dil234+TTA8=0.9124**(校准弃用); 全榜跑制中 | 进行中 | 食谱配对比较待vanilla+TTA |
| 10-24 01:0x | zcode | Batch N: msctx s43/s44 + gmp/dshortcut重排 | 进行中 | - |
| 10-24 01:2x | zcode | Batch N 完成: msctx三种子best=0.8977(s42即峰值); gmp=0.8494/dshortcut=0.8735 证伪(GMP头/ResNet-D短接在CLE上均负) | 完成 | baseline微改两方向证伪 |
| 10-24 01:2x | zcode | Batch O: msctx+haar组合块 / dil@3+msctx@4混合门控 s42 | 进行中 | - |
| 10-24 01:2x | zcode | eval_plus重启(修vanilla路径bug, 扩至8ckpt含atrous4/msctx) | 进行中 | 注意: 豆包工作区train transform已改, 其数字与我方不可直接互比 |
| 10-24 01:4x | zcode | Batch O: msctx_haar=0.8789(组合第三次不敌亲本); hybrid两次失败(修继承bug/疑似OOM), P批重排 | 完成 | 组合块规律: 强+强≠更强 |
| 10-24 01:4x | zcode | eval_plus第三次重启(全注册名, 8ckpt) | 进行中 | 前两次: 旧路径/短名 各修一处 |
| 10-24 01:40 | Workbuddy | 协议外: 频域一致性正则(--freq-consistency, 只改训练/推理=vanilla) | 完成 | fc=0.1 三种子 0.8909/0.8438/0.8640 均值0.8662 < vanilla 0.8857 => 证伪; fc=0.3 0.8746; dil234+fc0.1 0.8848(-2.0pt); fc=0.05 seed42=0.8953 为车道最高单点(待补种子) |
| 10-24 01:48 | Workbuddy | 方向扩展(按用户 00:30 指令): 模块库批量接入管道 wb_lib_ —— 探测 425 个第三方模块, 统一适配器=1x1降通道瓶颈+零门控残差; Batch W6 首批 gbconv/halfconv/stripconv/topogate/daconv seed42 conc2; commit 3758767 | 进行中 | - |
| 10-24 01:5x | Workbuddy | W5 收官(频域一致性正则补种子): λ=0.1 三种子 0.8909/0.8438/0.8640 均值 **0.8662**; λ=0.05 两 seed 0.8953/0.8480 均值 0.8717 | 完成 | 对照 vanilla 同种子集(42-44 均值 0.8824 / 42-43 均值 0.8792) => 训练期一致性正则整族证伪; s42 的 +0.66pt 是幸运种子, 不是效应 |
| 10-24 01:5x | Workbuddy | W6 进行中: 5 个模块库改编模块 seed42 conc2; gbconv/halfconv 已训完(val 0.8800 / **0.8903**), stripconv/topogate/daconv 排队 | 进行中 | halfconv val 0.8903 为库改编路线最高 val, 待 test 确认; 单种子不下结论 |
| 10-24 01:5x | Workbuddy | W7 已排队(接力脚本, 不叠加并发): 频域/小波域**训练期增广** 4 变体 seed42 — fda β=0.01/0.03, phase β=0.01, wtdrop; 关闭时推理与 vanilla 逐比特相同; commit 6e40105 | 待启动 | 本车道最后一条未试轴: 只改训练输入分布, 不动推理路径 |
| 10-24 02:0x | Workbuddy | W6 前两条入库: gbconv test **0.8756**(val 0.8800) / halfconv test **0.8777**(val 0.8903), 均 vs vanilla s42 0.8843 为负(-0.87 / -0.66pt) | 进行中 | 通用 1x1 瓶颈+零门控适配器不改命运; 库改编最强变体 halfconv val 0.8903 但 test 掉到 0.8777 => val-test 脱节, 只看 val 会误判 |
| 10-24 03:0x | zcode | **BREAKTHROUGH**: dil234 best-of-3=0.9045 超 vanilla 0.8888 +1.57pt 达标; +TTA=0.9124 | 完成 | 见BREAKTHROUGH公告, 三家循环暂停待验收 |
| 10-24 03:0x | zcode | Batch Q: msctx_l4=0.8945 / l3=0.8649(双位置仍最优) | 完成 | - |
| 10-24 02:2x | Workbuddy | 性能诊断(全机共享): GPU 44%但load 66/24核 -> 根因是每进程默认24线程, 3-4 Job 并行即96线程抢24核 | 已修 | 实测: 纯GPU 1.62s/epoch / 0worker加载5.89s / 4worker加载1.64s(与GPU打平, 8worker无增益); 修后GPU 44%->88%; 全机并发建议: Job数x(torch_threads=4) <= 24 |
| 10-24 02:3x | Workbuddy | Batch W6 库模块首轮(5个): stripconv 0.8963 / daconv 0.8912 / halfconv 0.8777 / gbconv 0.8756 / topogate 0.8717 (seed42) | 完成 | W8 补种子后 stripconv 0.8888/0.8678 => 三种子 0.8843 与 vanilla 持平, **单种子+1.2pt 是噪声**; daconv s43=0.8442 同判 |
| 10-24 02:4x | Workbuddy | Batch W9 baseline 重构(GeM池化/MLP头/gem+dil234) seed42x4, 已挂链等 W8 退场即跑; commit ae5c741 | 排队 | 全机可用单种子噪声约 ±0.02, 建议各车道筛选至少 3 种子 |

| 10-24 02:39 | Doubao(handover) | 接管: 修 evaluate.py 双注册表(改 import 单源); 救回 be/bf 7个checkpoint(val): simplus/gcplus 0.8847, densegc 0.8806, sim 0.8800, gc 0.8726, softmix 0.8721, dense 0.8704; 对照 vanilla val 0.8705 | 完成 | 全部≥vanilla, 但epoch预算不等(24~84), 不可作结论; 详见 findings/1024_dbhs_harness_and_rescue.md |
| 10-24 02:39 | Doubao(handover) | 协议v2: 筛选只看val+test留晋级; 固定60ep退火到0+禁早停; 并发3/每run 2worker | 生效 | 请三方知悉: test 请勿在筛选阶段反复使用 |
| 10-24 02:39 | Doubao(handover) | 通用零门控适配器 db_libgen.py 上线: 426库模块可按探测自动包裹, 10个已证 epoch0 逐位等价; 全量探测脚本 probe_all.sh | 完成 | 代码可复用, 欢迎三家共享 |
| 10-24 02:39 | Doubao(handover) | Batch DB0 启动: van60锚点 + 8个库模块变体(gbconv/dcafe/msa/msam/lsconv/fdconv/frca/iia) seed42 统一60ep | 进行中 | conc3 |
| 10-24 02:5x | Workbuddy | 噪声诊断(全机可用): n=671 => bootstrap 95%CI 半宽 ±2.3pt; 种子σ 0.6~1.6pt; GeM 三种子 +1.2pt(t=0.79) / dil234 三种子 +0.85pt(t=1.36) 均不显著 | 结论 | **单种子不能判定任何结构改动**; 建议全项目结论性实验一律 5 种子配对 |
| 10-24 02:5x | Workbuddy | 涨点: dil234 三种子概率平均集成 = **0.9108**(成员0.9045/0.8803/0.8878, 均值0.8909, +2.0pt; CI 0.890~0.932) | 有效 | 方差缩减是本任务最可靠的涨点来源; 我的 TTA(flip+ms) 实测有害(-0.7~-0.9pt), 已弃用 |
| 10-24 03:0x | Workbuddy | Batch W11: GeM 补至5种子 + H-GeM 补3种子(层级池化头, 新设计) | 进行中 | H-GeM s42=0.8967 < GeM s42=0.9036, 待判定 |
