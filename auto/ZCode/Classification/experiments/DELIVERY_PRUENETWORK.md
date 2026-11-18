# prueNetwork 全面更新交付记录 (2026-11-03)

## 1. CLE-Net 更新为审计后最终版
- 代码: convnext_cle2.py 两处 md5 一致(9247438e), 无需改; 差异在文档叙事
- 创新说明.md/README.md 重写: 最终形态 convnext_egb_scg(0.9630±0.0053, 配对+1.41pp 3/3);
  HSC 否决裁定(修复版组合-0.64pp, 仅消融表); 修复前数字作废声明; TCGA 域特异例外(hsc 85.52 > egb_scg 84.02)并报

## 2. experiment_data/ 分发(5项目×5文件)
- 各项目: metrics_<net>_all_datasets.csv(逐种子+均值) + tcga_all_networks.csv(11模型TCGA全域)
  + backbone_comparison_{cle_18networks, crossdomain_10backbones, cle_scidata}.csv(3份主干横评)
- 数据源: SF paper_data / DSG paper_data / CLE-Net sci_paper_data(审计14表) / ZCode TCGA(2026-11-02/03)

## 3. 重要更正
- EF-ConvNeXt 在 TCGA 从未实验(用户11-02指示暂缓); 此前对话误将其CUB结果73.28引用为TCGA
  → 已从5份TCGA表删除EF行, EF项目内写占位更正说明
