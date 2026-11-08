# GEM-ConvNeXt 交付记录 (2026-11-08)

## 交付位置
/home/login/data/cle/auto/prueNetwork/GEM-ConvNeXt/ (316KB 纯代码+数据包, 无权重)

## 内容
- 网络本体: models/optimize/convnext_cle_v5.py (convnext_v6f 最终网络 + convnext_v2m 消融档
  + 全部研究变体工厂) + models/Convnext.py 基座
- 训练/评测框架 8 文件(train.py 的 models.FasterNet 模块级导入加 try/except 保护, 发布包自洽)
- experiment_data/: cle_main_results.csv(4配置×3seeds×9指标+MEAN) / crossdomain_ablation.csv
  (5域×4配置) / model_complexity.csv(独占GPU口径) / negative_screening_summary.csv(25候选)
  / run_meta/(最终网络3seeds训练审计) / _build_data.py(数据溯源脚本)
- 文档: README.md / 创新说明.md / 运行与测试说明.md; demo.py 结构自测通过(CPU);
  包内全链路(导入+构建+严格加载审计+前向)验证通过

## 校验
- convnext_cle_v5.py 与源仓库当前版本一致(逐字节); train_cle_fair.py 含 _inflate_fgs/白名单/ACM/EMA
- 最终网络成绩: CLE 96.29±0.23(κ+2.09/恶性sens+2.46); 跨域 CUB+2.55/TCGA+1.33/heel+0.79/covid+0.21/BUSI-0.48
