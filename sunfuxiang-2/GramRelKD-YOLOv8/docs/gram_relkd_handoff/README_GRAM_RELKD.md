# Gram RelKD 研究快照交接索引

本目录是 `/data/cle/sunfuxiang-2/ultralytics-main` 在 2026-08-01 冻结出的完整研究交接材料。副本的正式主方法固定为 **Gram RelKD（Multi-scale Channel Gram Relation Distillation）**。

优先阅读顺序：

1. `NEW_CHAT_HANDOFF.md`：新对话所需的自包含上下文。
2. `METHOD_GRAM_RELKD.md`：方法和数学定义。
3. `MODEL_ARCHITECTURE_CHANGES.md`：L1-96 与 SR1-P4Only 结构演进。
4. `MODEL_CARD_GRAM_RELKD.md`：正式 test 指标、复杂度和限制。
5. `EXPERIMENT_LEDGER.md`：历史实验与路线状态。
6. `REPRODUCE_GRAM_RELKD.md`：复现、评估与隔离检查命令（仅文档，未执行训练）。
7. `PATH_MANIFEST.json`、`COPY_VERIFICATION.json`、`COPY_REPORT.md`：路径与复制证据。

醒目标记：

> FINAL MAIN METHOD = GRAM RELKD

保留的 MTRD、GAN、C2f_SPP、C2f_DSM 和夜间组合蒸馏代码仅用于历史证据与消融，不是最终推理模型的一部分，也不应在新对话中默认升级为主方案。

