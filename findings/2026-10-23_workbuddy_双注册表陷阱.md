# 新模型必须同时注册进两处 MODEL_REGISTRY(否则训练完评估必失败)

**发现人**: Workbuddy | **时间**: 2026-10-23 23:2x | **严重度**: 高(静默浪费整轮训练)

## 现象
本仓库存在**两套彼此独立**的模型注册表, 极易只改一处:

| 位置 | 用途 | 条目数 |
|---|---|---|
| `cle_fair_common.py` -> `MODEL_REGISTRY` | `train_cle_fair.py` 训练时解析 `--model` | 52 |
| `evaluate.py` -> `MODEL_REGISTRY`(文件内自带) | `evaluate.py` 评估时解析 `--model` | 38 |

`evaluate.py` **不 import** `cle_fair_common`, 两表各自维护, 已发生漂移。

## 后果
只往 `cle_fair_common` 注册新模型 -> 训练能正常跑完 300ep, 随后 runner 调
`evaluate.py` 时抛 `未知模型: xxx`, 整轮训练白跑, results.csv 只留一行 error。
(历史上 `swin_tiny` / `densenet121` 的 train_failed 与此类问题同源)

## 正确做法
新模型注册时, **两个文件都要加**, 同一行内容:
```python
"wb_xxx": ("models.optimize.wb_freq", "wb_xxx"),
```
另需在 `train_cle_fair.py` 的 `MODEL_ALLOWED_MISSING_SEGMENTS` 加白名单段,
否则加载 ImageNet 预训练时 strict 审计会因新增参数报 mismatch。

## 自查(一条命令)
```bash
cd $你的工作区 && /home/login/miniconda3/envs/sunfuxiang/bin/python -c "
import re,sys
a=set(re.findall(r\"\\"([a-z0-9_]+)\\":\s*\(\\"models\.\", open(\"cle_fair_common.py\").read()))
b=set(re.findall(r\"\\"([a-z0-9_]+)\\":\s*\(\\"models\.\", open(\"evaluate.py\").read()))
print(\"仅训练注册:\", sorted(a-b)); print(\"仅评估注册:\", sorted(b-a))"
```

## 附带: 已知坏条目
`evaluate.py` 里 `mobilenet_v2 -> models.MobileNetV2`, 但仓库中**无此文件**
(只有 MobileNetV3.py / MobileNet.py), 调用即 ModuleNotFoundError。建议不要使用该条目。
