# CLE-Net 修复版重跑 + codex 答疑 — 交接文档（新窗口用）

> 写于 2026-10-27。**主战场是 4090**（Dell-T2, RTX 4090 48G），n1(A800) 不用管。
> 本文档自包含：新窗口无需看历史对话，读完即可执行。

---

## 0. 当前状态（已完成）

三个实现 bug 已修复并提交到 4090 git（commit `015d087`，诊断依据 `GATE_DIAGNOSIS.md`）：

1. **零门控死锁**：SCG/HSC 分支含可训练参数，但被 `g≈0` 的标量门屏蔽 → 梯度链乘 g → 分支收不到梯度 → 停在随机初始化 → 输出噪声 → 门控梯度正负抵消永不打开。实测 30 个 ckpt 全部 `|g|≤4e-3`，且把门控强制置 0 后 test F1 **逐位不变**（off==orig），证明旧 CLE-Net/S-Net 功能上 ≡ ConvNeXt-T。
2. **HSC 伪注意力**：原实现把 768 维 reshape 成 (8,96) 后 `q@kᵀ` 得 **(B,8,8) 头间矩阵**，既非通道注意力也非标准 MSA。
3. **叠加覆盖 bug（codex Q01）**：`_swap` 两次整 stage 替换，后装的 SCG 覆盖了同 stage 的 EGB/ESA → 旧组合网络里 EGB/ESA 实际只剩 stage1，"渐进叠加"是假象。

**修复内容**（`models/optimize/convnext_cle2.py` / `convnext_cle5.py` / `train_cle_fair.py`）：
- SCG：去掉标量门，`StripBranch` 末层 `c3` 零初始化（ReZero 式渐进解锁，初始恒等）；
- HSC：重做 SE 式通道门控 `fc1(768→192)→GELU→fc2(192→768)`，`fc2` 零初始化，`f'=f⊙(1+tanh z)`；
- 新增 `CLENetBlock`/`SNetBlock` 统一叠加块，`CLENet`/`SNet` 改为按标志位在同块内叠加 → 现 **CLE-Net = EGB@stage1,2,3 + SCG@stage2,3 + HSC** 真叠加；
- `train_cle_fair.py` 优化器分两组：backbone(lr=1e-4, wd=5e-2) + **gate(egb_g/esa_g 标量门, lr=1e-2, wd=0)**。

n1 上已跑通 `smoke_fix.py`：8 个变体前向 `max|diff|=0.00e+00` 严格等价 baseline，覆盖断言全过。

---

## 1. 环境速查

| 项 | 值 |
|---|---|
| 服务器 | `ssh ubuntu-server` = login@192.168.1.160 |
| 项目根 | `/home/login/data/cle/auto/Network/CLE-Net` |
| python | `/home/login/miniconda3/envs/sunfuxiang/bin/python`（torch 2.12+cu130） |
| 数据 | `datasets/cle_ac_v1/{train,val,test}` = **2077/427/576** 张，三类 Benign/Malignant/body_of_stomach_normal（test 逐类 208/176/192） |
| 预训练 | `convnext_tiny_1k_224_ema.pth`，sha256 `14f3164e3ea6...`，路径先 `ls` 确认（run_meta 记为 `/data/cle/sunfuxiang/classifyML/pth/...`） |
| 训练脚本 | `train_cle_fair.py`；评测 `evaluate.py`；调度参考 `experiments/runner.py` 与 `experiments/jobs_*.json` |

## 2. 资源铁律（必须遵守）

- **单卡串行**，每次启动训练前 `nvidia-smi` 确认空闲；
- `num_workers≤3`、`OMP_NUM_THREADS=2`；
- **严禁 `kill PID 187555`**（服务器上别人的任务）；只管理自己启动的进程；
- 每次对项目的改动必须 `git add -A && git commit`（已配 safe.directory；提交身份用 `-c user.name=login -c user.email=login@wb.local`）。

---

## 3. 执行任务（按序）

**Step 1 — 4090 冒烟**（确认修复在 4090 可建、前向等价）：
```bash
cd /home/login/data/cle/auto/Network/CLE-Net
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv   # 确认空闲
OMP_NUM_THREADS=2 /home/login/miniconda3/envs/sunfuxiang/bin/python smoke_fix.py
# 期望末行 FORWARD_EQUIV_ALL_ZERO + SMOKE_OK
```

**Step 2 — 重跑修复版关键变体**，3 seeds(42/43/44)，用**新 save-dir `weights_cle_fixed_v1`**（勿覆盖旧 `weights_cle_ac_v1`）：
变体清单：`convnext_egb, convnext_scg, convnext_hsc, convnext_esa, convnext_egb_scg, convnext_scg_hsc, convnext_egb_scg_hsc(CLE-Net), convnext_esa_scg_hsc(S-Net)`。
单 run 命令模板：
```bash
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
PTH=<先 ls 确认的 convnext_tiny_1k_224_ema.pth 绝对路径>
cd /home/login/data/cle/auto/Network/CLE-Net
OMP_NUM_THREADS=2 $PY train_cle_fair.py --model convnext_egb_scg_hsc --num_classes 3 --seed 42 \
  --weights $PTH --data-path-train datasets/cle_ac_v1/train --data-path-val datasets/cle_ac_v1/val \
  --device cuda:0 --save-dir weights_cle_fixed_v1 --epochs 300 --early-stop 30 --batch-size 32 --lr 0.0001
```
训练只出 train/val 选 best ckpt；**test 评测单独跑** `evaluate.py --model <m> --weights <best.pth> --data-path datasets/cle_ac_v1/test`。建议用 runner 串行排队（先 1 个 run 确认无崩溃+门控被打开，再批量）。

**Step 3 — 验证门控这次真的打开了**：重跑后对 `weights_cle_fixed_v1` 跑 `probe_gate.py`（门控应明显 >1e-2）和 `gate_intervene.py`（**off 应 ≠ orig**，证明模块有实际贡献）。

**Step 4 — 汇总新消融表**：mean±std（sample std, ddof=1，由原始 results.json 未舍入值聚合，最后才舍入 4 位）。与 baseline（convnext_tiny mean 0.9476±0.0094）对比。

**Step 5 — 回答 codex 14 问**：读 `CODEX_QUESTIONS_FOR_WORKBUDDY.md`，新建 `WORKBUDDY_REPLY_TO_CODEX.md` 逐条答复（要点见 §4）。

**Step 6 — 更新简报 + 给 codex 方法说明**：更新 `CODEX_BRIEFING.md` 的错误项（§5 速查），出准确 Methods 素材。

---

## 4. codex 14 问答复速查（新窗口据此写 WORKBUDDY_REPLY_TO_CODEX.md）

> 标注【重跑后定】的条目，等 Step 2/4 出数后再填最终值；其余现在即可答。

- **Q01 覆盖bug**：确认是**实现 bug 非设计意图**。已修复为真叠加（EGB@1,2,3+SCG@2,3）。论文 Methods 按修复后写。**现有消融表（附录A）作废**，用修复版重跑数据。
- **Q02 HSC**：确认 8×8 是伪实现。已重做 SE 式通道门控（结构见 §0.2）。论文按新实现写。
- **Q03 EGB 数学**：按源码如实写——float32 FFT、高斯低通 `r=0.1·min(H,W)/max(H,W)`、分母是**方差**(非标准差)+1e-5、注意力 (B,C,H,W) 逐通道、融合 `sigmoid(低频sigmoid项+高频非负项)`。**收窄"零参数"**：每块仍有 1 个可学标量门（可为负）。边缘/噪声分离写为**设计动机**，非已证机制（无可视化证据）。
- **Q04 数据量**：确认 **train2077/val427/test576**，恶性 train=**371**。简报 3080/593/578 是把合计误写成 train。论文统一 2077/427/576。
- **Q05 归一化**：确认 CLE 主实验用 **ImageNet 统计量**（mean 0.485/0.456/0.406, std 0.229/0.224/0.225），因加载预训练权重走 ImageNet 分支。heelBone 的 run_meta 在 n1（`n1:/home/u2518483080/Classification`），需查后再裁定跨域归一化是否一致。
- **Q06 resize 差异**：协议事实。训练内 val 用方形 `Resize((256,256))+CenterCrop(224)`；`evaluate.py` 的 val/test 用短边等比 `Resize(256)+CenterCrop(224)`。分别列出，说明对比模型共享同一 test 评测脚本；CSV 的 val_f1 来自训练后短边评估，≠训练选 ckpt 用的方形 val macro-F1。
- **Q07 统计口径**：统一「原始 JSON 未舍入 → mean / sample std(ddof=1) → 最后舍入 4 位」。**区分 std 与方差**：CLE-Net 相对 baseline std 降 **52.97%**、方差降 **77.88%**，不混称。
- **Q08 结论边界**：【重跑后定】。修复前，"协同最优/方差最小/全场最高"不可用于论文。EGB 单点 mean 0.9593 > CLE-Net 0.9553 要如实写。S-Net best 0.9669 限定"本次指定比较集合内"。"synergy over additivity" 降为解释假设。
- **Q09 heelBone**：原始文件在 **n1**：`/home/u2518483080/Classification/experiments/HEELBONE_泛化实验报告.md`、`results_heelbone.csv`、各 run 的 `results.json`/`run_meta.json`。是**从 ImageNet 权重独立训练相同结构**（非迁移/零样本）。"零重调参"=不改 lr/epochs/增强/深度。S-Net best+0.78pp 但 mean 低 0.66pp，只能报**潜力与种子敏感性**，不能称均值全面占优。
- **Q10 A800 冲突**：优先 CSV 明确类别字段（sens_benign/malignant/normal）。FINAL_REPORT(0.9833) vs COMPARE_TABLE(0.9662) 的 BA 冲突需查原始 results.json 裁定。EGB"5路并发污染"归因是**假设非定论**，标"可能受负载/随机性影响"。
- **Q11 错误分析**：限定到具体模型/种子/类别，不写"各指标同步提升""所有模型12-18张"。
- **Q12 参数量**：用实际参数总数。**注意修复后变了**：HSC 改 SE 后 CLE-Net 从 33.67M 降到 **31.60M**（n1 冒烟实测 egb_scg_hsc=31,600,818）。THOP 漏计 layer_scale/自定义 LN（差 21408），需注明口径差异。
- **Q13 协议**：按实际写——AdamW lr1e-4 wd.05 bs32 ≤300轮 CosineAnnealing(T_max=300) 无warmup unweighted CE(label_smoothing=0) AMP+GradScaler，val macro-F1 选优，30轮早停。**声明固定 test 参与了开发反馈**（架构探索/补种子/冲刺），作为局限；不写跨硬件逐位确定；种子 std ≠ 95% CI。
- **Q14 证据**：数据画像脚本/患者划分/参考文献需补。PFESA(MICCAI)/StripConv(AAAI)/SCSA 出处**需核实**，不凭缩写生成参考文献；未核实处标 [待核对]。

---

## 5. 简报需修正的错误项（Step 6 用）

| 项 | 简报错误 | 正确值 |
|---|---|---|
| 数据量 | train 3080 / val 593 / test 578 | **train 2077 / val 427 / test 576** |
| 恶性样本 | 626（实为合计） | train=**371** |
| 归一化 | "数据集自算 mean/std 非 ImageNet" | **ImageNet 统计量**（加载预训练权重分支） |
| EGB 位置 | stage 1/2 | **stage 1/2/3** |
| ESA 位置 | "与 EGB 同位" | **stage 1/2**（与 EGB 不同位） |
| HSC 注意力 | "96 维通道 token 间 (B,8,96,96)" | 旧:8×8 头间矩阵；新:SE 式通道门控 |
| 叠加 | "渐进叠加 A→A+B→A+B+C" | 旧实现覆盖 bug；新实现真叠加 |
| 参数量 | CLE-Net 33.7M | 修复后 **31.6M** |

---

## 6. 关键文件指针（4090 项目内）

- `GATE_DIAGNOSIS.md` — 三个 bug 的完整诊断证据
- `CODEX_QUESTIONS_FOR_WORKBUDDY.md` — codex 的 14 问
- `smoke_fix.py` — 修复后冒烟（Step1）
- `probe_gate.py` / `gate_intervene.py` — 门控验证（Step3）
- `experiments/results_cle_ac_v1.csv` — 旧结果（参考，修复版勿混用）
- `models/optimize/convnext_cle2.py` / `convnext_cle5.py` / `train_cle_fair.py` — 已修复代码（commit 015d087）
