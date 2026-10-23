# Doubao 心跳

| 时间 | 批次 | 状态 |
|---|---|---|
| 10-23 23:10 | doubao_ab | 3/4完成(288=0.8776/ls05=0.883/384=0.8925, 均seed42筛选); ls10训练中; 单种子不下结论 |
| 10-23 23:21 | doubao_ab完成/batch_bc启动 | ab四seed42已同步doubao.csv; bc=@224复现对照+p50, conc2 |
| 10-23 23:26 | batch_bc | 启动冒烟通过: @224rep(默认patience30) + p50(--early-stop 50) 均seed42训练中, 透传已核对, 显存正常 |
| 10-24 00:13 | doubao_bc完成/batch_bd启动 | bc两行已同步doubao.csv; bd=vanilla_s42锚点+dil234_s43, conc2 |
| 10-24 00:16 | bd训练中/校准证伪 | vanilla_s42锚点ep22+dil234_s43 ep21; logit bias校准5/5 test降,已证伪留finding |
| 10-24 00:28 | 转发用户最高指示 | 性能优先/大胆突破; 公告已落 announcements; Doubao下批起转向db_全新结构族 |
10-24 00:46 | doubao_be 5任务排队(2在跑3待) | bd完成 vanilla_s42=0.8867/dil234_s43=0.8866; 5个db新模型冒烟全过(epoch0与vanilla等价)已启动 |
10-24 00:53 | doubao_be 5任务(conc2)+bf 2任务(conc1) 共7任务在跑/并发3 | GPU利用率99%/13GB; 7个db模型strict-load+epoch0与vanilla等价全过; bd完成vanilla_s42=0.8867/dil234_s43=0.8866 |

| 10-24 02:39 | 接管豆包 | 修复evaluate双注册表+救回7个白训ckpt(val 0.8704~0.8847) | 协议v2(仅val筛选/60ep退火) | DB0批9任务启动 | GPU 100% |

| 10-24 03:08 | 接管豆包 | 判决: 加性门控死(0.1~2%)/乘法门控活(2~27%) | 既往40个结构结论作废 | DB0/DB1已出7条 | s43/s44配对确认已排队 |

| 10-24 03:24 | 暂停 | 5个训练进程+2个runner已SIGSTOP冻结(GPU显存仍占7.9G) | 已完成3个批次的val筛选 | 等用户重新梳理后决定恢复/改向 |
