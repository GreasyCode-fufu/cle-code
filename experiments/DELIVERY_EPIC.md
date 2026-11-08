# 交付记录 — EPIC-MobileNet（暂定名, R6 轮, 2026-11-09）

**发布包**: `/home/login/data/cle/auto/prueNetwork/EPIC-MobileNet/`（396K; 网络本体 + 训练评测框架 + 门控/阶段 2 工具 + experiment_data + 三文档 + demo/selftest）

## 网络
MobileNetV3-Large（IMAGENET1K_V1）+ ① 证据保持冻结 /2–/16（门控决定保留深度）+ ② 解耦凸证据头（阶段 2，C 仅 val 选）
+ ③ FOV 感知四尺度证据。部署结构 2.975M 参数（−29.3%），MACs 0.232G，CPU 单线程 +1.8%。

## 主要结果
- 主域 cle_ac_v1（5 seeds）: 94.16 → **96.51（+2.35 F1）**, κ +3.51, MCC +3.44, AUROC +0.46, 恶性敏感度 +6.70;
  官方 3 seeds 96.43（+2.35）; 5/5 seeds CI > 0, 配对 t p = 3.4×10⁻⁵; 漏诊恶性 63 → 18（5 seeds 合计）。
- 跨域（3 seeds, 门控选档）: BUSI +0.51 / heelBone +4.79 / covid +0.33 / CUB +4.90 / TCGA +1.45 —— 连同主域 6/6 为正。
- 可迁移性门控（预注册 commit 3572063 → 73d1430 → 3bffd50）在 TCGA 样本外检验中预测成立; 6/6 域选中三档最优档。

## 包内验证（运行与测试说明.md §8）
selftest 通过; demo 通过; 包内 epic_refit → evaluate 端到端复现 seed42 test 97.07（与仓库逐位一致）;
包内 epic_gate 复现 CLE gap 0.61 → 启用 ①; 包内 train_cle_fair 严格加载 unexpected=0。

## 仓库内对应文件
- 代码: `models/optimize/mbv3_cle_r6.py`, `models/timm_light.py`, `experiments/r6_refit.py`, `experiments/r6_transfer_probe.py`
- 分析: `r6_p0_analysis.py` / `r6_p0b_pooling.py` / `r6_p0c_precise_bn.py` / `r6_p0d_refit.py` / `r6_p0e_combo.py` / `r6_significance.py`
- 文档: `R6_DESIGN.md`（初版设计）/ `R6_REPORT.md`（全过程与结果）/ `R6_MODULE_SCREEN.md` / `R6_LITERATURE.md` / `R6_GATE_PREREG.md`
- 数据: `results_cle_ac_v1_r6.csv`, `results_xd_{busi,heel,covid,cub,tcga}_r6.csv`, `weights_*_r6/`（eval 输出; 权重 .pth 不入 git）
- 生成 experiment_data: `python experiments/r6_build_data.py <输出目录>`
