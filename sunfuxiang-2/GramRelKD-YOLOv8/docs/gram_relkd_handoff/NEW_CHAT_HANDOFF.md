# Gram RelKD 项目完整交接

> **FINAL MAIN METHOD = GRAM RELKD**

新对话必须以此为固定结论：MTRD、GAN、C2f_SPP、C2f_DSM 和夜间组合 KD 是保留的对照/消融/失败路线，不得因为源码仍存在就默认将其重新设为主方案。

## 1. 用户研究目标

在香港墙体缺陷检测中，用训练期知识蒸馏帮助轻量 SR1-P4Only Student 恢复严格定位与弱类性能，同时保证部署模型不增加参数、GFLOPs 或 Teacher 依赖。后续重点应是 Gram RelKD 的机制、消融、多 seed、跨数据集、可视化和论文写作，而不是无边界叠加模块。

## 2. 项目、环境与数据

- 项目根目录：`/data/cle/sunfuxiang-2/GramRelKD-YOLOv8`
- Python：`/home/login/miniconda3/envs/yolo/bin/python`
- 外部数据集：`/data/cle/sunfuxiang-2/datasets/hongkongScale`
- 数据 YAML：`ultralytics/cfg/datasets/coco8.yaml`（文件名是历史遗留，实际内容指向 Hong Kong 数据）
- 类别：spalling、crack、moisture
- 规模：约 5,525 图像和 16,806 框；train/val/test=3315/1105/1105 图像
- 类别不平衡：crack≈81.6%，moisture≈6%
- 环境：Ultralytics 8.4.84、Python 3.12.13、torch 2.4.1+cu121、RTX 4090

## 3. 结构演进

原始 YOLOv8n 在本项目的金字塔约为 64/128/256，记录约 3.14M Params、8.3 GFLOPs、test mAP50-95≈0.429。

L1-96 将 backbone layer 6、head layer 12/18 的关键 P4 输出由 128 调整为 96，保持 P3=64、P5=256，形成 64/96/256。L1-96 是 Teacher：fused 2,641,337 Params、约 7.2 GFLOPs、test mAP50-95=0.429、mAP75=0.460。

SR1-P4Only 是 Student：保持外部 64/96/256，只将 layer 12 和 18 的 C2f hidden 48→40，layer 21 的 P5 hidden 保持 128。Student 为 fused/unfused 2,603,129/2,607,913 Params、7.044096 GFLOPs。SR1-P4Only 改变推理结构；Gram RelKD 不改变推理结构。

## 4. Teacher、Student 和关键路径

- Teacher YAML：`ultralytics/cfg/models/hongkong/L1-96.yaml`
- Teacher best：`runs/detect/runs/detect/hongkong_L1-96/weights/best.pt`
- Student YAML：`ultralytics/cfg/models/hongkong/yolov8n-L1-96-SR1.yaml`
- Gram best：`runs/detect/hongkong_SR1_P4Only_RelKD/weights/best.pt`
- Gram last：`runs/detect/hongkong_SR1_P4Only_RelKD/weights/last.pt`
- Paired NoKD best：`runs/detect/hongkong_SR1_P4Only_PairedNoKD/weights/best.pt`
- MTRD best：`runs/detect/hongkong_SR1_P4Only_MTRD/weights/best.pt`
- 正式方法代码：`ultralytics_ext/relation_kd.py`、`feature_hooks.py`、`kd_trainer.py`
- 历史正式入口：`scripts/distill_train.py`（保留源绝对路径）
- 副本安全入口：`scripts/distill_train_clone.py` 与 `scripts/run_gram_relkd_from_clone.sh`

所有绝对路径及存在性/哈希见 `PATH_MANIFEST.json`。

## 5. Gram RelKD 数学定义

正式名称是 Multi-scale Channel Gram Relation Distillation。在 layer 15/P3、18/P4、21/P5 上提取 Teacher/Student 同形状特征。每尺度转 float32、pool 到 20×20、展平为 `X∈R^(B×C×400)`、沿空间中心化并 L2 归一化，构造 `G=XX^T∈R^(B×C×C)`。Teacher detach，Student 保留梯度。

`L_l = mean_B(||G_s-G_t||_F²/C)`，`L_Gram=(L_P3+L_P4+L_P5)/3`，`L_total=L_detection+0.0431658384×ramp×batch_size×L_Gram`。warmup 为 10 epochs。

Teacher 必须 eval、冻结、no_grad，不进入 Student optimizer、EMA 或 checkpoint；推理只保留 Student。

## 6. 正式训练配置

epochs=300、batch=32、imgsz=640、workers=8、device=0、seed=0、deterministic=True、amp=True、optimizer=auto、patience=0、mosaic=1、mixup=0、close_mosaic=10、pool_size=20、P3/P4/P5 权重 1/1/1。Student 从 YAML 随机初始化，不载入历史 Student、Teacher、yolov8n.pt 或 SPP/DSM 权重。

## 7. 正式结果

Gram RelKD test：P=0.749849、R=0.675435、mAP50=0.704466、mAP75=0.459626、mAP50-95=0.435680；spalling/crack/moisture AP50-95=0.582001/0.410681/0.314357。

复杂度：fused/unfused 2,603,129/2,607,913，GFLOPs=7.044096，best.pt≈5.227 MiB。推理独立于 Teacher 与 `ultralytics_ext`。

## 8. 严格配对证据

Paired NoKD test：P=0.76923、R=0.66821、mAP50=0.69747、mAP75=0.43523、mAP50-95=0.42120；三类 AP50-95=0.56010/0.41213/0.29136。

Gram 相对 NoKD：mAP50 +0.00700、mAP75 +0.02440、mAP50-95 +0.01448、spalling +0.02190、crack -0.00144、moisture +0.02299。两次初始化哈希完全一致，seed、数据、超参数和调用链一致，唯一优化变量是 kd_weight。因此，在当前固定 seed 条件下，收益可归因于 Gram 关系蒸馏。

## 9. 已放弃或次优路线

- C2f_SPP≈0.420、C2f_DSM≈0.422：被 SR1-P4Only 支配。
- MTRD=0.432186：相对 NoKD 有效且 moisture 较好，但整体低于 Gram；仅作消融。
- E04 GAN=0.420200：低于 NoKD，判别器单边，无证明收益。
- 夜间最佳 E06 Gram+ResponseKD=0.429919：crack/moisture 有补充改善，但整体低于 Gram。
- E07 未启动。

## 10. 当前代码与副本状态

源项目被完整复制，runs、权重、日志、分析报告和自定义代码均保留；数据集未复制，因为它在项目根目录之外。复制前后原始快照的文件/目录/字节/路径和关键 SHA256 完全一致。新增交接文档与 clone-safe 脚本只存在于副本。Ultralytics 核心源码未为交接任务修改，本任务未启动训练或完整 test。

历史脚本中的源绝对路径被原样保留。继续研究时使用 clone-safe 新入口，不要批量替换核心源码。

## 11. 不得误改的约束

1. FINAL MAIN METHOD 必须保持 Gram RelKD，除非未来有预注册、严格配对、多 seed 证据明确推翻。
2. Teacher 只训练期存在，不得进入 checkpoint 或推理。
3. 新方法不得增加最终 Student Params/GFLOPs，除非用户明确改变研究目标。
4. 不得用 test split 反复调参；先用 val 选择，再一次性 test。
5. fused 与 unfused Params 必须分开注明。
6. 单 seed 结果不得表述为统计显著。
7. 不得自动启动实验；先提出假设、对照、预算和停止规则。

## 12. 后续优先问题

优先级建议：多 seed 严格配对；P3/P4/P5 单层与 leave-one-out；center on/off；Gram 与等梯度尺度 Feature MSE；mAP75/IoU 分段机制；crack/moisture 的类别机制；关系矩阵可视化；外部数据集；训练成本与边缘设备延迟。完整问题表见 `OPEN_RESEARCH_QUESTIONS.md`。

## 13. 结果解释限制

数据类别不均衡、框可能宽松、图像尺寸大且仅单数据集；正式提升不是每类一致，Precision 下降、Recall 上升，crack 轻微下降。任何论文 claim 都应限定为此任务、此 Student/Teacher 和当前确定性训练证据。

## 14. 术语表

- **Gram RelKD**：多尺度通道 Gram 关系蒸馏。
- **L1-96**：P4 输出为 96 的非均匀通道 Teacher。
- **SR1-P4Only**：只压缩两个 P4 C2f hidden 的 Student。
- **Paired NoKD**：调用链相同但 kd_weight=0 的严格对照。
- **MTRD**：多步 Markov 转移关系蒸馏，次优消融。
- **fused/unfused Params**：卷积/BN 融合后与融合前的参数量，不能混用。
- **training-only overhead**：Teacher/KD 只增加训练成本，不影响部署图。

## 15. 新对话开始方式

上传本文件并使用 `NEW_CHAT_START_PROMPT.txt`。先要求新对话核对 `PATH_MANIFEST.json`、`MODEL_CARD_GRAM_RELKD.md` 和 `COPY_VERIFICATION.json`，再讨论任何新研究。

