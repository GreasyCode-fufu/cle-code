# 新任务：A–C限定数据集与网络优化，版本1

用户明确取消旧完整数据任务，要求只取原train、val、test各自A–C图像作为新数据集，在该数据集上优化设计模型。旧R15取消，已完成项和所有历史阴性结果保留。本协议替代旧任务目标，不把完整原test强制作为本任务主要评价集。

## 数据构建与一次冻结

保持原患者互斥的train2386/185组、val538/39组、test671/41组划分，各split在自身内部独立患者SGKF5×42/43/44分级。分级模型为同一ImageNet预训练ResNet34，60epoch/batch32/原256-224增强，AdamW1e-4wd.05cosine0/AMP，无早停。每外折只在其rest患者中以内层SGKF10选最高macroF1的checkpoint，first-valid-three-class内层规则；每图每seed恰一次整患者留出预测。三个原split之间不混用训练图或标签。

原train的15个有效签名OOF模型和预测完全符合此规则，直接复用；仅新增val/test各15模型，共30新分级CNN，不重复有效训练。每split均先得到全部图片三预测，再按同一原A–E规则：correct_rate为现有标签正确率、confidence为现有标签平均概率；A两者≥.95，B两者≥.85，E任一<.30，D任一<.60，否则C。A/B/C全部保留。原始标签不改，不因后续分类器的预测修改等级或清单。

生成三个split的完整等级CSV、原图索引/患者/类别/哈希、A–C清单、D/E排除清单、类别与患者保留率；一次冻结datasets/cle_ac_v1链接目录及dataset_manifest.json/dataset_config.json。所有后续候选共用同一清单。链接只引用原图，原图不移动、不覆盖。原3595图cache复用，物理含全部split；各分级loader仅访问其指定split。

该构建定义了模型一致性筛选后的A–C研究人群，结论限于此数据分布；等级不是人工病理真值。测试子集定义使用了test内部的交叉拟合与现有标签，这属于预先固定的回顾性数据构建，不声称新的未经任何处理的外部临床验证。该说明不改变用户指定的A–C主要目标。

## 模型优化与设计

数据冻结后，立即在A–C train训练普通ResNet34三seed42/43/44基线，A–C val选checkpoint与后续结构方案；A–C test为固定最终测试集，各网络统一评价。基线优化器/60epoch/batch32/224输入与上述相同，推理不需要分级元信息。大规模结构搜索不反复看test：结构候选与消融用A–C val，最后冻结候选后统一测固定A–C test。

后续结合shared/module_library做机制明确的局部结构、忠实原方法与参数/计算控制，逐轮不超过6方案×3seed。新数据分布上的结论重新验证；旧任务的阴性结果保留作背景，不自动等同于本A–C数据任务结果，也不把模块组合直接称为新颖性证明。

## GPU/CPU高效调度

资源实测为约49GiB GPU、24CPU逻辑核、约128GiB RAM。初始最大6训练任务并行，每任务CPU计算线程1、原DataLoader2 workers、pin_memory/persistent_workers/预取与已存在mmap图像cache；不让大量解码争用CPU。训练模型初始化后将CPU线程降为1，不改GPU模型/批量/增强参数。预检查保存初始化/AMP等价与实际data-wait/step-time/峰值显存基准。

任务完成立即补队列，0.5秒轮询，GPU/CPU资源采样约2秒；显存预留6GiB、每新任务至少2GiB启动余量，CPU负载过高时限制继续加任务，保留其他Agent进程。并发只影响执行调度，不按中途性能选择样本/模型。保存resource_samples.csv和每任务耗时、GPU利用率，后续基于实测调整调度而不是盲加并发。有效签名checkpoint复用，后处理失败不从头重训。

## 核验与交付

核验45分级模型来源(15复用、30新增)、60epoch/患者fit-inner-outer互斥/逐图预测/现有标签不变；每split每图3预测、所有等级与A–C清单独立复算、原患者划分不交叉、原图SHA、链接目标、各类别非空。数据冻结审计通过后才进入基线。归档配置/代码/协议/分级/保留率/基线与结构结果、图表和中文说明到本聊天outputs/CLE_AC_dataset，权重与大图cache留服务器。有效操作和结果Git提交，保持login归属。
