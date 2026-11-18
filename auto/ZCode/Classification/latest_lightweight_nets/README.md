# 最新轻量网络对比代码库（R6, 2026-11-09）

每个子文件夹为一个网络的**官方代码库**（git 浅克隆，`fetch_latest.sh` 重新获取；精确 commit 与许可证见 `MANIFEST.tsv`）。
代码只作为模型定义来源（官方实现零改动），训练/评测在本仓库统一协议内进行
（AdamW 1e-4 / wd 5e-2 / cosine / bs32 / 相同增广 / val Macro-F1 选点 / 早停 30 / seeds 42–46 / evaluate.py 9 指标）。
适配器：`models/latest_light.py`；结果：`experiments/R6_REPORT.md §15`、`results_cle_ac_v1_r6.csv`。

| 子文件夹 | 网络 / 出处 | 对比档位 | 官方 ImageNet-1k top-1 | 权重来源（下载方式） |
|---|---|---|---|---|
| `LSNet_CVPR2025` | LSNet, "See Large, Focus Small", CVPR 2025 | LSNet-T（非蒸馏） | 74.9 | HuggingFace jameslahm/lsnet（经 hf-mirror 直连） |
| `iFormer_ICLR2025` | iFormer, ICLR 2025 | iFormer-T（300e） | 74.1 | GitHub release v0.9（直连） |
| `EfficientViM_CVPR2025` | EfficientViM, CVPR 2025 | EfficientViM-M1（e300） | 72.9 | Google Drive（代理） |
| `SHViT_CVPR2024` | SHViT, CVPR 2024 | SHViT-S1 | 72.8 | GitHub release v1.0（直连） |
| `MobileMamba_CVPR2025` | MobileMamba, CVPR 2025 | **未纳入对比** | — | Google Drive（代理） |

- **MobileMamba 放弃原因**：官方代码依赖需编译的 selective-scan CUDA kernel；未编译时纯 PyTorch 回退实测 >30 min/epoch（224²，4090），
  训练不可行（用户 2026-11-09 确认放弃）。其隐式依赖 PyWavelets 隔离安装于 `_pydeps/`（不污染 conda 环境）。
- **权重转存**：官方发布文件以受限反序列化（只放行张量重建类）转存为纯张量 state_dict，存于
  `/home/login/data/cle/sunfuxiang/classifyML/pth/latest_r6/`；严格加载审计 unexpected=0、仅分类头缺失。
- **兼容性说明**：LSNet 按 timm 0.5 写法（`hf_hub` 配置字段、`default_cfg` 入参）在 timm 1.0 下报错，适配器在加载点打兼容补丁；
  LSNet 的动态核为 Triton 实现，**仅 GPU 可运行**（无 CPU 部署路径，CPU 延迟记为 NA）。
- **网络下载**：服务器经代理访问 GitHub/HF CDN 偶发 SSL 失败；直连 GitHub release 与 hf-mirror.com 可用，Google Drive 走代理可用。
