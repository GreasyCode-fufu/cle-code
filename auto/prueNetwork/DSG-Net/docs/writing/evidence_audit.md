# TCGA 证据核对记录

只读核对时间：2026-11-04。未训练、未修改远端任何文件、未扫描原始图像。

## 已保存材料

- 用户指定的4个CSV已原样下载。
- results_tcga_ep.csv：ZCode实际调度器的逐种子结果（含明确seed列）。
- tcga_dsg_seed42_run_meta.json：实际DSG-Net训练元数据。
- tcga_dsg_seed42_val_report.md / tcga_dsg_seed42_test_report.md：实际验证/测试报告。用户禁则要求本轮不引入逐类指标，此处仅作为大小及结果来源核对资料。
- model_complexity_bench.json：GPU独占统计的原始汇总；未包含硬件型号、数值精度、预热细节。

## 运行与数据划分

实际run_meta为num_classes=6、img_size=224、seed42、ImageNet归一化、val_macro_f1选点、无权重交叉熵。预训练SHA256为14f3164e3ea6ac32ab3f574f528ce817696c9176fad4221e0a77a905a7360595。

data_path_train=/data/cle/auto/datasets/TCGA/train；data_path_val=/data/cle/auto/datasets/TCGA/val。

实际验证报告明确10634张，测试报告明确5702张。进一步按主agent授权对实际训练目录只读计数（未打开图像）：train=36377（全部png）、val=10634、test=5702；seed43和44日志也与seed42一致，均显示验证10634、测试5702。结果与用户MD一致、与消息摘要36277/10636冲突。证据已发送主agent；实际计数和三种子日志片段保存于actual_split_counts_and_logs.txt。

源码预处理：train_cle_fair.py:715起，Resize((256,256))→随机水平/垂直翻转→±15°随机旋转→RandomCrop224→ToTensor→ImageNet Normalize；cle_fair_common.py:223起验证/测试统一Resize((256,256))→CenterCrop224→Normalize。这是网络输入尺寸；不证明所有原始文件均为300×300。

分组策略可引用用户MD描述：TCGA按patient ID，FIMM按slide ID，分组不可拆，目标70/20/10，分组内类别计数驱动贪心+局部优化，种子42，多次确定性初始化。但未独立审计此次训练路径对应分组清单，不宜直接声称全部患者独立性已核验。FIMM仅slide独立也不等于patient独立。

## 主结果与主干横评

results_tcga_ep.csv确认seed42/43/44：baseline 83.81/84.73/84.61；DSG-Net 84.30/85.29/86.10。原始横评CSV f1_seeds列是按分数排序的显示序列，不可据其顺序进行跨模型配对。

横评名为10backbones，但表内共10行，包含baseline，即9个其他主干。若加DSG-Net则11行；不得虚构缺失的ResNet34统计。

FasterNet汇总显示84.43±0.97，baseline84.38±0.41；显示均值差0.05，但CSV vs_tiny列为0.04（先对未显示精度求差后舍入可能导致）。建议不使用此列进行新配对推断，正文表述“约+0.05pp的均值偏移”。

DSG accuracy CSV为88.62；按用户后续答复使用CSV。Macro-Specificity不在用户所列四CSV中，若需用97.76/97.62，来源应标用户给定摘要；不虚称四CSV本身包含该列。

## 参数与效率

model_complexity.csv表注：THOP MACs，224×224，batch1；参数包含nc=3分类头；FPS为GPU空闲独占PyTorch eager、batch1、100次均值，测量日期2026-11-03。

因此27.84M是该CSV的三分类统一计数口径，不应称为TCGA六分类完整参数的精确逐参数计数。与之前文稿直接sum(parameters)统计存在计数方法差异，不能混为一张无说明的表。

同一结构文档明确新增38592(+0.14%)，0.07%没有支持依据。MACs表中4.45→4.47为两位小数记录，不能从此恢复精确增幅；可沿用用户给定约0.4%但须表明近似来源，不与此前主要算子计数0.30%混淆。

GPU记录：814 FPS / 1.23ms（baseline）与509 FPS / 1.97ms（DSG）。没有记录GPU具体型号、精度、同步方式和预热次数，未找到专用统计脚本。现有单次evaluation_report的FPS另有测量值，不应用来替换用户指定独占测量509/814。

## 可支持表述范围

可写“本文呈现的四个任务均获得正的平均配对增益”，不能据当前资料写“项目唯一未出现负增益域的网络”：缺乏完整、同批、同协议其他模型四域面板；且现有设计文档保留其他域的负结果。稿件按用户限定范围不应扩大到项目全部历史。

±0.74为总体标准差，不是方差也不是置信区间，并且大于baseline±0.41，不能写“比基线更稳定”或“方差极小”。三个种子增益全正和跨种子绝对分数波动是不同性质。

Macro-AUROC下降但硬标签指标上升支持“改善体现在当前分类决策”，不证明概率校准变好或某类解剖结构成为原因；近98不构成排序能力已无改进余地的证据。

OCA数学量是方向最大占比及其标准化，不是完整熵倒数；DOSE复合条带不是独立±45°核，也不旋转采样网格。继续以旧稿代码准确表述为准。
