# 当前实验：五网络 × BUSI/CUB × 三种子
用户最新决定：放弃Covid，只比较BUSI和CUB。总任务数30（5×2×3）。原始文件夹名称保留以免影响正在运行的程序。
模型：MambaOut-Tiny、TransNeXt-Tiny、MambaVision-T、ThinkingViT、CPUBone-B2。种子42/43/44。不新增CLE-Net或baseline，用户自行整理CLE-Net。
数据只读：ZCode/Classification/datasets/busi_v1和cub_v1。BUSI 546/118/116，3类；CUB 5194/800/5794，200类。dataset_manifest.json为当前两个数据集的逐图SHA256。精确跨集合重复审计见cross_split_duplicate_audit.json，该检查不证明患者/病例独立。
环境为cle_topnets（torch2.5.1+cu124），源码在code/，环境及版本/权重哈希分别见environment.json、pip_freeze.txt、source_sha256.json和pretrained_sha256.json。五个模型的3类/200类预训练严格加载均已通过，preflight_*.json记录核验。
固定公平协议：224输入、ImageNet初始化/归一化、AdamW lr1e-4 wd0.05、bs32、cosine300、patience30、AMP、未加权CE。按val Macro-F1选权重，训练结束后评估固定test。ThinkingViT保留既有3H→6H、threshold0、最终阶段CE适配；CPUBone为bfrobust变体。
run_experiment.py为入口。status.json为实时状态，coordinator_busi_cub.log为当前调度日志。runs/BUSI、runs/CUB为当前实验。TRAIN_DONE表示正常训练结束；只有eval_test/results.json成功生成才计为完成。
per_seed.csv、summary.csv、summary.json只汇总BUSI/CUB并增量更新。n<3仅为中间结果。最终必须30/30完成，每配置3种子，样本标准差ddof=1。
数据集各自独立训练，是跨数据集适用性比较，不是CLE_ac训练后直接外部测试。与旧CLE-Net结果对比前须核对环境和协议，不得跨环境拼表宣称公平优势。
Covid的104组跨划分重复、用户先前保留划分选择及后续放弃决定见SCOPE_CHANGE.json、abandoned_covid_history。runs/Covid及历史目录中的结果均不计入最终表，不得恢复或继续该数据集。
旧MobileNet-V3迁移实验也保持用户明确停止状态，不得自动恢复。
每次子任务前检查GPU，如有其他任务则等待，不终止其他Agent进程。失败或中断不得伪报完成；有中断checkpoint无TRAIN_DONE时需检查并记录恢复处理，不能静默当作完整训练。

## 最新用户指令：停止全部实验
当前与后续队列均已停止。不得自动恢复。BUSI完成15/15，CUB仅部分完成；不得把CUB单种子或中断结果写为完整3种子汇总。保留所有已完成测试及中断权重、日志。实时终止记录见status.json。
