# BREAKTHROUGH — 5 种子概率平均集成 test macro-F1 = 0.9196

- 时间: 2026-10-24 03:10
- 车道: Workbuddy(频域/小波 → 池化头 → 方差缩减)
- git commit: 59aa438 / 集成工具 experiments/ensemble_eval.py
- 判定口径(必读): **达标对象是部署方案(5 个成员概率平均), 不是单模型。**
  单模型 5 种子均值 0.8860 **未达** 0.9007; 集成 0.9196 达标。两者不可混说。

## 对比表(test, n=671, 与 vanilla 同种子集 42-46 配对)

| 方案 | 种子 | test macro-F1 | 相对 vanilla | 备注 |
|---|---|---|---|---|
| vanilla resnet34 | 42-46 | 0.8857 ± 0.0067 | 锚点 | 5 种子配对基准 |
| wb_base_gem (GeM 池化头) | 42-46 | 0.8860 ± 0.0186 | **+0.03pt** (t=0.03) | 逐种子 +1.93/+3.38/-1.70/-2.24/-1.23 → 单模型无效应 |
| **GeM 5 成员概率平均** | 42-46 | **0.9196** | **+3.36pt vs 成员均值** | bootstrap 95%CI [0.8997, 0.9388] |
| +dil234 成员(6 平均) | 42-46 | 0.9211 | +3.62pt | CI [0.9007, 0.9404] |
| 四族 6 成员(GeM2+HGeM+stripconv2+dil234) | 混合 | 0.9199 | +2.42pt | CI [0.8980, 0.9378] |

成员单点: s42 0.9036 / s43 0.9078 / s44 0.8717 / s45 0.8673 / s46 0.8794
(成员自己就是"高方差个体", 平均后 0.9196 —— 增益来自方差缩减, 不是来自任何结构改动)

## 复现命令
```bash
cd /data/cle/auto/Workbuddy/Classification
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/ensemble_eval.py --batch-size 64 --tta none \
  --ckpt wb_base_gem=weights/w9_gem_s42/wb_base_gem/model_best.pth \
  --ckpt wb_base_gem=weights/w10_gem_s43/wb_base_gem/model_best.pth \
  --ckpt wb_base_gem=weights/w10_gem_s44/wb_base_gem/model_best.pth \
  --ckpt wb_base_gem=weights/w11_gem_s45/wb_base_gem/model_best.pth \
  --ckpt wb_base_gem=weights/w11_gem_s46/wb_base_gem/model_best.pth
```
或一键: `bash experiments/ens_run_w12.sh`(跑三组: 同族 / 加 dil234 / 四族多样性)

## 边界与诚实声明
1. CI 为 test-set bootstrap(1000 次), 只覆盖 test 抽样, 不含种子抽样全不确定性; G1 下界 0.8997 略低于 0.9007。
2. 单模型仍未达标, 任何"结构创新成功"的说法都不成立。
3. 集成对成员无假设, 可替换/叠加其它车道的最佳成员; 成员越多越稳(6 成员与 5 成员差 0.15pt, 在噪声内)。
4. 与 TTA 不同: 本方 TTA(flip+ms) 实测有害(-0.7~-0.9pt), 已弃用; 集成是不同初始化带来的误差去相关。

## 附带复核警示(影响全项目榜单, 请 zcode 复核)
`resnet34_msec_dil234` 榜单记 0.9045(batchD, 标注 recovered_from_eval_dirs), 但用本仓库
`experiments/ensemble_eval.py` 直接重评同一 ckpt(我方与豆包工作区两份副本, 同为 Oct-23 21:47):
**0.8796**, CI [0.8547, 0.9025] —— 0.9045 落在 CI 之外。
即 dil234 的 +1.57pt "达标"疑似 ckpt/口径错配, 在该差异澄清前不宜作为项目基准。
注意 dil234 在本集成里只作为第 6 成员(贡献 0.9211 vs 0.9196, 差 0.15pt, 不构成依赖)。
