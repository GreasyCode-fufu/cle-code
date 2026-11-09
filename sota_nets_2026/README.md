# R6-B 对比实验 — CLE-Net vs 最新顶会网络（2026-11-09）

独立任务（与 Codex 的 `prueNetwork/CLE-Net/latest_top_networks` 半成品任务无关，网络选择互不重叠）。

## 任务
以本仓库统一协议（AdamW lr=1e-4 wd=5e-2 / cosine / bs32 / 224² / 相同增广 / val Macro-F1 选点 / 早停 30 / seeds 42–46），
在 cle_ac_v1（train/val/test = 2077/427/576）上同批训练 **CLE-Net**（`convnext_egb_scg`，ConvNeXt-T + EGB/SCG 零初始门控；
本仓库 `models/optimize/convnext_cle2.py` 已确认与发布包逐字节一致）与 4 个最新顶会网络，对比 9 指标与复杂度。

## 对比网络（均为官方代码零改动 + 官方 ImageNet-1k 权重；子文件夹 = 各自代码库）
| 子文件夹 | 网络 | 出处 | 档位 | 参数(3类) | 权重来源 |
|---|---|---|---|---|---|
| Jumbo_ICLR2026 | Jumbo | ICLR 2026 "Thicker and Quicker: A Jumbo Token for Fast Plain ViT" | small | 64.2M | HF antofuller/jumbo（用户手动上传） |
| GmNet_ICLR2026 | GmNet | ICLR 2026 "Revisiting Gating Mechanisms From A Frequency Perspective" | s3 | 7.4M | 官方 Google Drive npy（用户手动上传） |
| iFormer_ICLR2025 | iFormer | ICLR 2025 | S | 6.2M | GitHub release v0.9 |
| LSNet_CVPR2025 | LSNet | CVPR 2025 "See Large, Focus Small" | S | 15.7M | HF jameslahm/lsnet |
（Codex 任务已选 MambaOut/MambaVision/ConvNeur/CPUBone/TransNeXt，本任务全部避开；Mamba 类按用户指示不需要。）

## 适配说明（models/sota2026.py）
- 合成包加载官方文件（支持相对导入，不改官方代码）；timm≥0.9 兼容补丁（LSNet 的 hf_hub/default_cfg 旧写法）。
- **GmNet 特别处理**：官方发布 checkpoint 含全部 `f1.bn`/`dwconv2.bn` 键（各 17 块），与仓库工厂默认 `f12_bn=False,
  dwconv2_bn=False` 不符 —— 按 checkpoint 实际结构构建（`f12_bn=True, dwconv2_bn=True`，其余超参与官方 `gmnet_s3` 工厂
  逐项一致），严格加载审计 unexpected=0、仅 head 缺失。
- 权重经受限反序列化转存为纯张量（`pth/sota2026/`；GmNet 为 np.save 字典 → 张量重建）。
- 四网络分类头根名（head/classifier）均在审计白名单内。

## 运行
```bash
python experiments/runner.py experiments/jobs_MN.json 4   # 6 模型 × 5 seeds（含 ConvNeXt-T 同批基线）
```
结果: `experiments/results_cle_ac_v1_r6.csv`（batch=MN）+ `weights_cle_ac_v1_r6/<model>_seed*/.../results.json`；
汇总脚本 `experiments/r6_collect.py`。
