# 2026-10-24 豆包工作区接管: 双注册表修复 + 7个白训checkpoint救回 + 协议v2

**接手人**: Doubao(handover) | **状态**: 生效
**触因**: 豆包额度耗尽, 由接替者继续"炼丹"; 接手时发现 be/bf 两批 7 个模型
全部 `train 成功 / eval 失败`, results.csv 只留 error 行。

## 1. 根因: evaluate.py 的私有 MODEL_REGISTRY (即"双注册表陷阱"的第二次复发)
`evaluate.py` 用 `choices=sorted(MODEL_REGISTRY.keys())` 做 CLI 校验, 而它维护的
是文件内的**私有**字典。新模型只注册进 `cle_fair_common.py` 时, 训练能跑完,
评估在 argparse 阶段就被 `invalid choice` 拒绝 => 整批白训。

**修复(本次)**: `evaluate.py` 删除私有字典, 改为
`from cle_fair_common import MODEL_REGISTRY`。单源真相, 这类故障不会再复发。
> Workbuddy 工作区的 `evaluate.py` 仍有同样的私有字典, 建议同样改掉。

## 2. 救回 7 个 checkpoint (val macro-F1, 未评 test)
| 模型 | val | 相对 vanilla(val 0.8705) |
|---|---|---|
| db_simplus (软核选择) | 0.8847 | +1.42pt |
| db_gcplus (多路重参数) | 0.8847 | +1.42pt |
| db_densegc (block级密集门控) | 0.8806 | +1.01pt |
| db_sim (原版 SimConv) | 0.8800 | +0.95pt |
| db_gc (原版 GCConv) | 0.8726 | +0.21pt |
| db_softmix (核×膨胀软选择) | 0.8721 | +0.16pt |
| db_dense (block级密集门控-dwref) | 0.8704 | -0.01pt |

**但这批数字不可用于下结论**: 7 个 run 的 epoch 预算不等 (24~84 ep), 而 checkpoint
是按"val 最大值"选的 => 跑得越久/越抖的模型被 max 选中的机会越多, val_best 被系统性抬高。
=> 必须用等预算重跑才可比 (协议v2 的固定 60ep 就是为了消掉这个偏置)。

## 3. 协议 v2 (只改测量, 不改被测量的东西)
1. **筛选只用 val**; test 只在晋级决赛时评一次 (避免第 N 次偷看留出集)。
2. **固定预算**: 所有 run 都跑 60 epoch, `CosineAnnealingLR(T_max=60)` 退火到 0,
   `--early-stop 0` 关闭早停。原协议 `T_max=300 + patience30` 会让 run 在 ~44ep、
   lr 仍 9.4e-5 时被截断, 既压低绝对分数又放大方差。
3. 每条结果同时记录 `val_best` 与 `val_last`, 让"选模偏置"始终可见。
4. 并发 3 / 每 run 2 dataloader worker: 24 核被 7×(1+4) 个进程打满过一次(load 118),
   CPU 才是瓶颈, GPU 不是。

## 4. 通用零门控适配器 (models/optimize/db_libgen.py)
426 个库模块的构造签名与输出契约各不相同。做法: 对每个模块**探测**构造参数
(`(ch,)`, `(ch,ch)`, `in_dim=ch` ...) 与前向形状(4D / (B,N,C)), 形状不符时用 1x1 conv +
插值补齐, 然后统一套 `out = x + gamma*BN(branch(x))`, `gamma=0`。
- 已验证: 10 个库模块在 epoch-0 与 vanilla **逐位相同** (max|diff| = 0.0)。
- 探测结果写入 `experiments/lib_menu.jsonl` (含参数量的可用模块清单)。
- 全量探测: `experiments/probe_all.sh` (逐模块独立进程, 崩溃隔离)。
