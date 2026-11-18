# 项目背景

## 任务

本项目研究香港墙体缺陷目标检测，类别为 `spalling`、`crack`、`moisture`。数据集是副本之外的外部依赖：

`/data/cle/sunfuxiang-2/datasets/hongkongScale`

项目中的实际数据配置文件为 `ultralytics/cfg/datasets/coco8.yaml`。名称沿用 Ultralytics 历史文件名，但其有效内容已指向上述 Hong Kong 数据集的 train、val、test 子目录；不能根据文件名误认为 COCO8 示例数据。

## 数据规模和标注性质

- 总图像约 5,525 张，总标注框约 16,806 个。
- train：3,315 张，约 10,010 个框。
- val：1,105 张，约 3,553 个框。
- test：1,105 张，约 3,244 个框。
- 图像多数接近 4624×3472。
- `crack` 约占全部标注的 81.6%，`moisture` 约占 6%，类别明显不均衡。
- 部分框较宽松；moisture 框可能包含较多普通墙面；crack 通常细长；不同类别依赖不同尺度特征。

## 研究目标

在压缩后的 SR1-P4Only Student 上，利用训练期 L1-96 Teacher 的多尺度通道关系监督恢复或提高检测质量，同时保持部署时只有 Student：不增加推理参数、GFLOPs 或 Teacher 依赖。

## 固定环境

- Ultralytics 8.4.84
- Python 3.12.13
- torch 2.4.1+cu121
- NVIDIA RTX 4090
- Python 环境：`/home/login/miniconda3/envs/yolo`

精确包列表和 GPU/OS 快照见 `ENVIRONMENT_SNAPSHOT.md`、`requirements_snapshot.txt` 和（若成功生成）`conda_explicit.txt`。

## 证据边界

当前正式结论来自单一 seed=0 的确定性 300-epoch 配对训练。它支持“在该固定 seed、数据和训练链条件下 Gram RelKD 带来收益”，但不能声称跨随机种子的统计显著性、跨数据集普遍有效或每个类别都必然提升。

