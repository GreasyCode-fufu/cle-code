# 复现命令与模型使用

服务器ubuntu-server，项目 `/home/login/data/cle/auto/GPT`；专属实验目录 `experiments/ac_structure_20261024_window01`。Python固定 `/home/login/miniconda3/envs/sunfuxiang/bin/python`，Git固定 `/home/login/miniconda3/bin/git`。运行前读该目录PROTOCOL、config和status，确认本流水线是否存活；不要同时启动两个bootstrap。所有输出由login写入，历史目录/共享库不修改。

以下为Linux服务器内命令：

```bash
cd /home/login/data/cle/auto/GPT
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
STUDY=experiments/ac_structure_20261024_window01
# 已通过的配置和必要数值预检查，重跑前先检查仍存活的本流水线
$PY $STUDY/run_window01.py prepare
# 首次复现：预检、吞吐测量、正式训练由bootstrap串联。
# 正在运行时不要重复启动；bootstrap.lock和pipeline.lock会拒绝重复。
$PY -u $STUDY/bootstrap.py
```

源码初次冻结提交`0c6c084`；数值预检提交`859e056`；配置、jobs、原始清单及源码SHA在实验目录。后续运行/阴性/审计继续提交，不能以git reset覆盖历史。权威原始冻结AC清单为 `experiments/ac_dataset_v1/dataset_config.json` 和 `dataset_manifest.json`，不是重新扫描所有图得到新清单。

有效模型按签名复用。单任务例如：

```bash
$PY $STUDY/run_window01.py worker --job window01_relational_s45
```

如果完整训练和概率都存在且SHA/签名正确，该命令直接返回；若complete已存在仅推理没完成，只补严格加载和验证概率。若只有不完整训练日志，脚本拒绝覆盖，必须先核对具体日志与仍活进程。独立审计只能在所有任务完成后运行：

```bash
$PY $STUDY/audit_window01.py
```

审计重算21个模型（18新+3复用）全部427张val概率、checkpoint选择、指标、患者配对bootstrap及预算/所有权。它不会运行test。本任务没有提供开放的test搜索入口，最终一次test须先另行预注册冻结主要baseline、候选、全部checkpointSHA和比较规则。

权重位置：新模型在 `weights/ac_structure_20261024_window01/window01_<arm>_s<seed>/model_best.pth`，主要baseline在原 `weights/ac_backbone_recipe/acbackbone_dino_full_s<seed>/model_best.pth`。每个complete.json有完整SHA，种子必须45/46/47全部报告，不能把最高seed当结果。新增checkpoint均完整包含主干和读出；当前构建器沿用已冻结DINO初始化文件，因此在服务器直接运行不需再次下载预训练。移动到其它机器时还需同步源码、对应原公开预训练文件及固定环境；目前没有声称独立单文件部署已验证。

已完成的指定模型可作单图研究推理，不按最高seed自动选择：

```bash
$PY $STUDY/predict_window01.py --arm relational --seed 45 \
  --image /path/to/image.jpg --device cuda --output /path/to/prediction.json
# CPU使用--device cpu；CPU为FP32，CUDA为FP16，可能有小数值差异。
```

预处理：RGB→PIL双线性resize256×256→center crop224→ImageNet均值/标准差。类别固定0良性、1恶性、2正常胃体。概率未经临床校准，输出不是医生诊断。不可根据新test图错误继续调参。原始医疗图像和大型权重留服务器；本窗口交付源码、配置、报告、清单元数据、审计和逐图概率。

持续工作：本聊天专属heartbeat每15分钟衔接，本地automation id `cle`，正常训练和一般阴性不通知。首轮结束并不代表目标完成；如DINO饱和，按用户追加要求转向公平调优的其它竞争性主干，在其结构比较前重新冻结该阶段baseline，并保留所有阶段结果。最终test后不能依据结果再换baseline制造成功。
