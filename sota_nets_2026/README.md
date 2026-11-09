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

## 结果（2026-11-09, MN 批, 5 seeds, 统一协议; 详见 R6_REPORT §16）
| 网络 | CLE Macro-F1 | κ | 恶性敏感度 | 参数 | MACs | CLE-Net−该网络 |
|---|---|---|---|---|---|---|
| **CLE-Net (egb_scg)** | **95.58±0.15** | 93.42 | 93.30 | 31.3M | 4.85G | — |
| ConvNeXt-T（同批基线） | 94.76±0.79 | 92.16 | 91.70 | 27.8M | 4.45G | +0.82（4/5） |
| Jumbo-small | 94.05±1.45 | 91.16 | 88.64 | 64.2M | 4.78G | +1.54（3/5） |
| GmNet-s3 | 92.96±0.66 | 89.43 | 86.14 | 7.4M | 1.27G | +2.63（5/5） |
| iFormer-S | 93.68±1.53 | 90.48 | 89.89 | 6.2M | 1.08G | +1.91（5/5） |
| LSNet-S | 91.42±0.55 | 87.23 | 84.09 | 15.7M | 0.55G | +4.17（5/5） |

## 三 seeds 口径（42/43/44）: 见 R6_REPORT §16.1 —— CLE-Net 95.50; Jumbo-small 95.23(+0.26, 1胜1平1负, 基本打平); GmNet 92.61(+2.89, 3/3); iFormer-S 94.17(+1.32, 3/3); LSNet-S 91.62(+3.88, 3/3)
