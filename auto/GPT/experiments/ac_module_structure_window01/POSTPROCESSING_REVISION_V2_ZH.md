# 当前实际入口与后处理修订

本说明补充启动快照MODULE_STRUCTURE_AND_USE_ZH.md，旧文件/原配置保留。训练模型、jobs、seed、食谱、数据及训练引擎均未改。

当前协调入口是 **bootstrap_module_structure_v3.py**；查看bootstrap_identity_v3.json、bootstrap_status_v3.json、bootstrap_v3.log并核对/proc。v2等待协调器已经核对身份后终止，当时没有新模型worker；现有几何轮GPU worker继续完成。转换证据见COORDINATOR_TRANSITION_V3.json。

原因：成熟C4参照CSV用mean_p0/mean_p1/mean_p2，单视图用p0/p1/p2。原审计器的单视图读取接口会在将来的C4对照汇总时报错。修订仅新增audit_module_structure_v2.py；run_module_manager_v2.py在新进程内选择这个审计版本，再调用原冻结run函数。postprocessing_revision_v2.json同时锁原config SHA及新增源码SHA。原audit、训练源码与签名完全保留，不因后处理问题重训有效模型。

修订已实际回放5个对照组×3seed、每单元576图/41患者，共15个完整概率单元：检查原角与既有单视图相等、四角均值与C4 CSV相等、原登记SHA、标签/患者/图像身份、有限概率及独立整数Fraction F1。comparators_precheck_v2.json通过。此次只是重读已批准的既有开发test结果，不是新的网络前向、训练或新候选的独立确认。

现阶段MODULE_ROUND_REPORT_ZH.md/round_results.json仅包含这5类成熟参照。不要把参照预检查表解释成18个新增候选已经完成。

继续命令（仅确认无同身份协调器和worker存活后）：

```bash
cd /home/login/data/cle/auto/GPT
/home/login/miniconda3/envs/sunfuxiang/bin/python \
  experiments/ac_module_structure_window01/bootstrap_module_structure_v3.py
```

不要重复启动已存活v3，也不要恢复旧v2或单独调用原run入口来绕过已注册审计修订。v3会复用已通过的CPU/CUDA预检，等待原几何轮完整审计后，再隔离实测并发并训练18个注册候选。

另有方向机制证据experiments/ac_orientation_mechanism_evidence_window01：仅处理既有val概率，三种骨干×3seed，39患者条件bootstrap与独立Fraction/分组审计完成；C4平均在val新增错误多于纠正，不支持“无条件方向平均一定改善”的解释。该阴性诊断不改变已冻结候选，也不替代方向矩与普通旋转增强的配对重训。
