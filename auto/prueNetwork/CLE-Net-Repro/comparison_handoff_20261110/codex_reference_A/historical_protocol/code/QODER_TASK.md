# Qoder 任务书：4090 补齐 CLE-Net 论文消融实验

> 交接方：WorkBuddy（负责本项目全部网络设计与已完结实验的 agent），2026-10-27。
> 你的任务：在 4090 服务器上补齐论文所需的消融实验，使论文全部主表格可以统一以 4090 结果为准。
> 本文件是你的唯一任务规格来源；背景知识见同目录 `CODEX_BRIEFING.md`（必读第一节与第七节即可）。

---

## 一、项目位置与环境（4090 = Dell-T2）

- **项目根目录**：`/home/login/data/cle/auto/Network/CLE-Net`
- **登录方式**：本机若需 SSH，别名 `ubuntu-server`（即 `login@192.168.1.160`）；若你直接在该机器上有终端则无需 SSH。
- **Python**：`/home/login/miniconda3/envs/sunfuxiang/bin/python`（conda env `sunfuxiang`，torch 2.12.0+cu130）
- **git**：系统 git 在 `/usr/bin/git`（shell 的 PATH 可能不含，先 `export PATH=/usr/bin:$PATH`；miniconda 里也有 git）
- **数据集**：`datasets/cle_ac_v1`（项目内软链，train 3080 / val 593 / test 578，三分类）
- **预训练权重**：`pretrained/convnext_tiny_1k_224_ema.pth`（软链，消融全部只需要它）
- **结果 CSV**：`experiments/results_cle_ac_v1.csv`（runner 自动追加；现有 CLE-Net 6 种子、S-Net 5 种子、SCG 2 种子的记录，勿改动历史行）
- **对比数据源**：`/home/login/data/cle/auto/ZCode/Classification/experiments/results_cle_ac_v1.csv`（同机同协议的 11 个对比主干历史结果，已做过五项协议核验，可直接复用）

## 二、任务清单

### A 组（必做）：消融链补测，共 16 个训练 run

目的：论文消融表（baseline → +A → +B → +C → 两两组合 → 完整）目前在 4090 上缺中间环节。

| 模型注册名 | 说明 | 种子 |
|---|---|---|
| `convnext_egb` | 创新点 A 单点（PFESA 零参数边缘-结构门控） | 42, 43, 44 |
| `convnext_hsc` | 创新点 C 单点（通道自注意力校准头） | 42, 43, 44 |
| `convnext_egb_scg` | A+B 组合 | 42, 43, 44 |
| `convnext_scg_hsc` | B+C 组合 | 42, 43, 44 |
| `convnext_esa` | Sobel 显式边缘单点（S-Net 的 A 替换项，用于 EGB vs ESA 对照） | 42, 43, 44 |
| `convnext_scg` | B 单点补第 3 个种子（已有 42/43） | 44 |

**作业表文件**（写到 `experiments/jobs_qoder_abl.json`）：

```json
[
 {"model": "convnext_egb", "seed": 42, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_egb", "seed": 43, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_egb", "seed": 44, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_hsc", "seed": 42, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_hsc", "seed": 43, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_hsc", "seed": 44, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_egb_scg", "seed": 42, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_egb_scg", "seed": 43, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_egb_scg", "seed": 44, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_scg_hsc", "seed": 42, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_scg_hsc", "seed": 43, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_scg_hsc", "seed": 44, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_esa", "seed": 42, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_esa", "seed": 43, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_esa", "seed": 44, "pth": "convnext_tiny_1k_224_ema.pth"},
 {"model": "convnext_scg", "seed": 44, "pth": "convnext_tiny_1k_224_ema.pth"}
]
```

### B 组（可选）：补 densenet121 对比主干（3 runs）

A800 的 10 主干表里有 densenet121，4090 的 ZCode 复用表里没有。若想两机表格主干完全对齐可补。权重在 `/home/login/data/cle/sunfuxiang/classifyML/pth/densenet121-a639ec97.pth`，先软链：

```bash
ln -sf /home/login/data/cle/sunfuxiang/classifyML/pth/densenet121-a639ec97.pth pretrained/densenet121-a639ec97.pth
```

作业同格式（模型名 `densenet121`，pth `densenet121-a639ec97.pth`，种子 42/43/44）。注意：训练脚本已内置 densenet 权重键名重映射与白名单（本项目 train_cle_fair.py 已含该修复），无需改动。

### 不需要做的（明确排除）

- v2–v4 探索变体（os/dsh/pld 等）：论文仅作 Discussion 叙事素材，A800 数字够用，不必 4090 重跑；
- heelBone 跨域实验：数据在 A800 侧，作为独立的泛化章节保留 A800 结果即可（如需迁移 4090 再另行安排）；
- **严禁修改任何模型代码、训练脚本、评测脚本**——只允许新增 jobs json、运行、记录。若某变体加载失败（白名单报错），如实记录并停下报告，不要 hack 白名单。

## 三、执行规范（必须遵守）

1. **串行执行**：runner 并发参数固定用 `1`（单卡单任务，此前实测 GPU util 70–87%）。
2. **每次启动前检查**：
   ```bash
   nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l   # 应为 0
   cat /proc/loadavg                                                    # 1min load ≤ 8（Dell-T2 为物理机，多核）
   ```
   不满足则等待，不要叠加启动。
3. **启动命令**（在项目根目录）：
   ```bash
   export PATH=/usr/bin:$PATH
   git add experiments/jobs_qoder_abl.json
   git -c user.name=sunfuxiang -c user.email=sunfuxiang@dell-t2.local commit -m "Qoder 补实验: 消融链作业表(16 runs)"
   nohup /home/login/miniconda3/envs/sunfuxiang/bin/python experiments/runner.py experiments/jobs_qoder_abl.json 1 >> experiments/qoder_abl.nohup.log 2>&1 &
   ```
4. **监控**：`grep -a "^\[done\]" experiments/qoder_abl.nohup.log`，每 run 约 3–8 分钟，16 runs 预计 1.5–2 小时。单行格式如 `[done] convnext_egb seed42: ok val=0.97xx test=0.9xxx (X.Xmin)`。
5. **已知坑（重要）**：
   - **失败重跑前先清残留目录**：若某 run 失败，`weights_cle_ac_v1/<model>_seed<seed>/` 会留下半成品；重跑前必须 `rm -rf` 该目录，否则训练脚本会写到 `<model>-1/` 子目录、runner 找不到 ckpt（历史上踩过）。
   - runner 每 run 自动跑 val+test 评测并 append 到 CSV；`status=error` 的行不要删除，追加正确的行即可。
   - 评测走 `evaluate.py`，本项目 CLE 数据集无需 `--class-names` 参数（默认即正确）。
6. **git 留痕（硬性要求）**：对本项目的每次写操作（新增 jobs、修改报告等）都要 commit，message 说明改了什么为什么。纯查询不用。
7. 全程不要动 `weights_cle_ac_v1/` 下已有成功 run 的目录。

## 四、启动前冒烟测试（先做，全过再启动 A 组）

```bash
cd /home/login/data/cle/auto/Network/CLE-Net
/home/login/miniconda3/envs/sunfuxiang/bin/python - <<'EOF'
import sys, torch
sys.path.insert(0, ".")
from cle_fair_common import build_model
import train_cle_fair as T
sd = torch.load("pretrained/convnext_tiny_1k_224_ema.pth", map_location="cpu")
sd = {k: v for k, v in (sd.get("model", sd)).items() if not k.startswith("head.")}
CR = T.CLASSIFIER_ROOTS
for name in ["convnext_egb", "convnext_hsc", "convnext_egb_scg", "convnext_scg_hsc", "convnext_esa", "convnext_scg"]:
    m = build_model(name, num_classes=3).eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 224, 224))
    missing = [k for k in m.state_dict() if k not in sd]
    allowed = T.MODEL_ALLOWED_MISSING_SEGMENTS.get(name, set())
    bad = [k for k in missing if k.split(".", 1)[0] not in CR and not any(seg in allowed for seg in k.split("."))]
    assert not bad, (name, bad)
    print(name, "OK", tuple(y.shape))
print("SMOKE_ALL_OK")
EOF
```

期望最后一行 `SMOKE_ALL_OK`。任何断言失败 → 停止并记录，不要修改代码绕过。

## 五、验收标准与产出

1. A 组 16 行全部 `status=ok` 出现在 `experiments/results_cle_ac_v1.csv`（batch 列 = `qoder_abl`）；
2. 汇总一张**4090 消融总表**（test macro-F1，逐种子 + best + mean±std），行序：baseline(ZCode) → +A(egb) → +ESA(esa) → +B(scg) → +C(hsc) → A+B → B+C → CLE-Net(A+B+C, 已有 6 种子) → S-Net(ESA+B+C, 已有 5 种子)；写入 `reports/QODER_消融总表_4090.md`；
3. 与既有结论做一致性核对：SCG 单点此前 4090 为 0.9617(s42)/0.9326(s43)，A800 上 egb/hsc 单点偏弱（0.94–0.96）——新结果若出现大幅偏离（>±2pp），在报告中标注；
4. **回报文件**：写 `QODER_REPORT.md` 到项目根目录，内容：执行了哪些 run、逐 run 结果表、遇到的问题与处置、git commit hash 列表、给 WorkBuddy 的疑问（如有）；
5. 全部产出 git commit。

## 六、参考数字（判断结果是否"正常"用）

- 4090 已有：CLE-Net 6 种子 0.9474–0.9599；S-Net 5 种子 0.9198–0.9669；SCG s42/s43 = 0.9617/0.9326；baseline 5 种子 mean 0.9476±0.0094（best 0.9615）
- A800 参照（跨机绝对值有 ±1pp 差异，仅作量级参考）：egb 单点 0.9374/0.9582、hsc 单点 0.9401/0.9654、egb_scg 0.9602/0.9406、scg_hsc 0.9617/0.9471/0.9441、esa 单点 0.9601
- 合理预期：全部变体落在 0.92–0.97 区间；若出现 <0.90 先怀疑训练环境问题而非模型问题

—— 完 ——有任何与本任务书冲突之处，以不修改代码、不改动历史结果为原则自行判断；无法判断时在 QODER_REPORT.md 中列出问题并暂停。
