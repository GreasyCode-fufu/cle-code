# 完整复制与交接报告

## 最终结论

状态：**SUCCESS**  
FINAL MAIN METHOD: **GRAM RELKD**

## 复制范围与时间

- 源：`/data/cle/sunfuxiang-2/ultralytics-main`
- 实际目标：`/data/cle/sunfuxiang-2/GramRelKD-YOLOv8`
- 复制开始：2026-08-01T12:58:00+08:00
- 复制完成：2026-08-01T12:58:52+08:00
- 交接审计完成：2026-08-01T13:13:45.011524+08:00
- 方式：`rsync -aH --info=progress2,stats2`，未使用 `--delete`

## 原始快照一致性（新增交接文件之前）

| 项目 | 源 | 目标 | 结果 |
|---|---:|---:|---|
| 总文件数 | 2839 | 2839 | MATCH |
| 总目录数 | 386 | 386 | MATCH |
| 文件总字节 | 1077845555 | 1077845555 | MATCH |
| `du -sb` | 1077845555 | 1077845555 | MATCH |
| 相对文件列表 | — | — | MATCH |
| 空文件列表 | — | — | MATCH |
| 符号链接列表 | — | — | MATCH |
| 全文件 checksum dry-run | — | — | CLEAN |

交接材料加入后的目标当前统计为 2882 个文件、387 个目录、1078246416 文件字节；增加量全部来自交接文档与 clone-safe 辅助脚本。所有原始相对路径文件在最终审计时仍与源逐文件 checksum 一致。

## 关键 SHA256

| 相对路径 | 源 SHA256 | 结果 |
|---|---|---|
| `ultralytics_ext/feature_hooks.py` | `a4a6ebf490199175cca80b8b24d192461588e72a1aa6139619d2c18f53953ee7` | MATCH |
| `ultralytics_ext/relation_kd.py` | `193f8663570c8b248cb4d905bca05ec3e44ff49771dd9d03544d4ca1127892a1` | MATCH |
| `ultralytics_ext/kd_trainer.py` | `38b7fac21023b63e79f5d08326a4c64dea84ce0262d1693c67fa74f8d54f5086` | MATCH |
| `scripts/distill_train.py` | `e7f183ebf93b30cba17b9d977c5de7890b36a9d495d258a8935096563e7c4ed2` | MATCH |
| `scripts/calibrate_kd_weight.py` | `cd799bdf8cad856076a74217e75f8fd8d29aa3f8e72665e69b4f03bb95cea593` | MATCH |
| `ultralytics/cfg/models/hongkong/L1-96.yaml` | `4ad48daa00d17c698d3c90de40f97781d492bc7d8d68bc43020ed6ef67557adb` | MATCH |
| `ultralytics/cfg/models/hongkong/yolov8n-L1-96-SR1.yaml` | `25e4f80805be5853dc890d3828bad2bf1b8c6c7dcd4b1701a7fb0f8bee7b6dc2` | MATCH |
| `runs/detect/runs/detect/hongkong_L1-96/weights/best.pt` | `39b033ec1a18ea58dbc21d4e4bbcb2bf7426b84c172d461087416d201b51ef04` | MATCH |
| `runs/detect/hongkong_SR1_P4Only_RelKD/weights/best.pt` | `221304b847e9b900ccbef36723b27e72227d661279d735e131d0c60a41607ef7` | MATCH |
| `runs/detect/hongkong_SR1_P4Only_RelKD/weights/last.pt` | `a5df9b01346829241d41767319b4e97dd0bd5ce6f2640b6866174624fc18e55b` | MATCH |
| `runs/detect/hongkong_SR1_P4Only_PairedNoKD/weights/best.pt` | `6eed9dfa8c01861af6adf89d0b0f352db384795830d655b960d5dd530f68c9df` | MATCH |
| `runs/detect/hongkong_SR1_P4Only_MTRD/weights/best.pt` | `7f1a999b1377601c926f42fd17aa03176eefc902d45df758b44c48a03d57be00` | MATCH |
| `runs/analysis/nightly_kd_sweep_20260801/final_report.md` | `c478e48f8f9c26b41e171e83aa6d52273480495ede242aa2d32dc7c86772e955` | MATCH |
| `runs/analysis/kd_formal_result/final_assessment.md` | `2f3b13b097d9854ad22881ff0863135f44a18eafcc634ef90292987990925987` | MATCH |
| `runs/analysis/kd_formal_result/formal_test_results.json` | `2c63cfbbdd820a3c7d8873158773988ce888b0de520411b5a8dca64ff92f4add` | MATCH |
| `runs/analysis/kd_formal_result/kd_training_report.md` | `7aa96ad71e7d001c9860a6ef21aba70e04a9ff579a4cfe326de9875a38162769` | MATCH |
| `runs/analysis/kd_paired_nokd_result/paired_comparison_report.md` | `85155b89636ab02ce05e0ad083c9642a24447edebdcbb230e0634b987b09c7a2` | MATCH |
| `runs/analysis/kd_paired_nokd_result/paired_config_verification.json` | `d87f033f1e03d1b9871fb0c7888e5965cfc0be9e248bcc340d8a708b281e97c8` | MATCH |

## 未复制内容与外部依赖

唯一有意不复制的是数据集，因为其不在源项目根目录中。外部依赖为 `/data/cle/sunfuxiang-2/datasets/hongkongScale` 和 `/home/login/miniconda3/envs/yolo`；未复制 conda 环境。runs、所有历史 best.pt/last.pt、日志、分析报告、自定义 KD 代码和 YAML 均已复制。

## 最小运行验证

- Ultralytics 导入与版本：SUCCESS，8.4.84。
- Gram best.pt 加载：SUCCESS。
- SR1-P4Only：SUCCESS，hidden 40/40/128，P3/P4/P5 形状 64/96/256。
- 随机张量 eval 前向：SUCCESS。
- checkpoint Teacher/KD/判别器键：均不存在。
- 独立推理导入 `ultralytics_ext`：否。
- fused/unfused Params：2,603,129/2,607,913。
- GFLOPs：7.044096。
- 完整 test：未运行。
- 训练：未启动。

## 新增文件（只在目标副本）

- `GRAM_RELKD_PROJECT.md`
- `docs/gram_relkd_handoff/ABANDONED_BRANCHES.md`
- `docs/gram_relkd_handoff/CLONE_SMOKE_TEST.json`
- `docs/gram_relkd_handoff/COPY_CRITICAL_SHA256_DESTINATION.txt`
- `docs/gram_relkd_handoff/COPY_CRITICAL_SHA256_SOURCE.txt`
- `docs/gram_relkd_handoff/COPY_FILE_LIST_DESTINATION.txt`
- `docs/gram_relkd_handoff/COPY_FILE_LIST_SOURCE.txt`
- `docs/gram_relkd_handoff/COPY_REPORT.md`
- `docs/gram_relkd_handoff/COPY_VERIFICATION.json`
- `docs/gram_relkd_handoff/ENVIRONMENT_SNAPSHOT.md`
- `docs/gram_relkd_handoff/EXPERIMENT_LEDGER.md`
- `docs/gram_relkd_handoff/FINAL_INTEGRITY_AUDIT.json`
- `docs/gram_relkd_handoff/HARD_CODED_SOURCE_PATHS.txt`
- `docs/gram_relkd_handoff/METHOD_GRAM_RELKD.md`
- `docs/gram_relkd_handoff/MODEL_ARCHITECTURE_CHANGES.md`
- `docs/gram_relkd_handoff/MODEL_CARD_GRAM_RELKD.md`
- `docs/gram_relkd_handoff/NEW_CHAT_HANDOFF.md`
- `docs/gram_relkd_handoff/NEW_CHAT_START_PROMPT.txt`
- `docs/gram_relkd_handoff/OPEN_RESEARCH_QUESTIONS.md`
- `docs/gram_relkd_handoff/PATH_MANIFEST.json`
- `docs/gram_relkd_handoff/PROJECT_CONTEXT.md`
- `docs/gram_relkd_handoff/README_GRAM_RELKD.md`
- `docs/gram_relkd_handoff/REPRODUCE_GRAM_RELKD.md`
- `docs/gram_relkd_handoff/conda_explicit.txt`
- `docs/gram_relkd_handoff/requirements_snapshot.txt`
- `scripts/clone_smoke_test.py`
- `scripts/distill_train_clone.py`
- `scripts/run_gram_relkd_from_clone.sh`
- `scripts/test_relkd_clone.py`

此外，复制证据目录包含源/目标文件、目录、空文件、符号链接列表和 diff/checksum 证据。历史脚本未修改；21 处源绝对路径记录在 `HARD_CODED_SOURCE_PATHS.txt`，副本专用入口使用 clone-safe 路径。

## 安全与完整性

- 源项目未修改：True。
- Ultralytics 核心源码未修改：True。
- 原始复制文件最终仍逐文件相同：True。
- Teacher/Student/历史权重未修改。
- 训练未启动：True。
- 完整 test 未启动：True。

最终状态：**SUCCESS**
