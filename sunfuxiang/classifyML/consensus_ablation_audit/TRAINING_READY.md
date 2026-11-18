# CLE 共识消融训练就绪状态

```text
TRAINING_READY = TRUE
```

该状态表示代码、manifest、固定数据划分、预训练权重、元数据和统一汇总链路已通过非训练验证；本阶段仍未启动 18 个正式 GPU runs。

## 逐项回答

1. **manifest 功能是否实现？**  
   是。`train_cle_fair.py` 已新增唯一参数 `--train-manifest`，并在 Full train 扫描结果上过滤；完整校验逻辑位于 `consensus_manifest.py`。

2. **不传 manifest 时旧 Full 行为是否保持一致？**  
   是。修改前后均为 4977（Normal 1578、Benign 2136、Malignant 1263），有序路径/标签 SHA-256 均为 `2a365f3576326c3a40dadf9c0ab2d4f5ca4b2a8b42875445c56adeb75647f021`；旧 run_meta schema 也保持一致。

3. **六个 manifest 的数量与类别组成是否完全正确？**  
   是。R=3500、V=3474、C=3749、RV=3103、RC=3141、VC=3115，类别组成和上一轮审计完全一致；所有验证状态为 PASS。

4. **RVC 是否再次与现有 HQ 精确一致？**  
   是。computed RVC、权威 consensus CSV 和现有 HQ train 均为 2880，sample_id 双向差集均为 0；没有生成新的 RVC 训练 manifest。

5. **fixed val 是否仍为 538？**  
   是，且 manifest 与 val 的 sample_id 交集为 0。

6. **fixed test 是否仍为 671？**  
   是，且 manifest 与 test 的 sample_id 交集为 0。

7. **统一汇总脚本是否已经 dry-run 验证？**  
   是。已从显式锁定的六份旧 predictions.csv 统一重算全部要求指标和 per-class/confusion matrix，三 seed 使用 sample std、ddof=1；18 个尚未产生的新 predictions 被正确标记为预期缺失。

8. **18 个训练 run 是否已经可以安全启动？**  
   是，从代码和数据准备角度可以。固定协议如下：ResNet34、224、batch 32、max 300 epochs、AdamW、lr 1e-4、weight decay 0.05、CosineAnnealingLR(T_max=300)、未加权 CE、label smoothing 0、AMP、patience 30、按 val Macro-F1 选优、seeds 42/43/44。必须使用 `/home/login/miniconda3/envs/sunfuxiang/bin/python` 和当前 `/home/login/data/cle/...` 路径。

9. **如果还不能启动，剩余 blocker 是什么？**  
   没有代码或数据 blocker。真正启动前仍需做即时运维检查：确认共享服务器 GPU/显存空闲、没有同名 run 正在执行、`weights_consensus_ablation/<subset>/seed<seed>` 目标未被占用。这些状态会随时间变化，不构成本阶段实现 blocker。

## 启动边界

本轮没有运行 R/V/C/RV/RC/VC × seeds 42/43/44，没有创建 Random-2880，没有修改历史 Full/HQ 权重、预测或 summary。正式训练只能由后续明确指令触发。
