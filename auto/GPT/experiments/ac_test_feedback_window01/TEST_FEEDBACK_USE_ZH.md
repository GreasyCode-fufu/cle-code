# 逐方法test反馈执行与使用说明

当前流程采用用户再次确认的“train训练 → val最早最佳权重 → test测试 → test反馈设计下一方法”。参数梯度只使用原固定train；不改变患者split、标签、A–C等级或索引。test参与开发，因此不称独立泛化验证。历史查看test、A–C构建使用其标签等局限一并保留。

本次快照完整报告14组方法、39个不同checkpoint，每组seed45/46/47全部报告。ConvNeXt-Base224在两个阶段复用同一组权重和预测，所以方法数×3不等于不同权重数。DINO三seed旧test预测按checkpoint SHA和既有审计复用，又逐一独立复算确认；没有重复训练。14组、39个权重都通过完整576图/41患者的概率、索引、标签、状态和指标审计。

完整结果见TEST_FEEDBACK_REPORT_ZH.md、test_results.json；成本、同主干224/384对照和逐seed混淆矩阵见COST_AND_NEXT_EVIDENCE_ZH.md及costs_and_resolution_controls.json。逐图预测、原配置、方法登记、源码、旧失败记录和独立审计保存在服务器及证据包。主要B0尚需最后一组资格训练和整体审计，不能把当前DINO临时参照冒称最终主要baseline。当前没有正式训练H1，也没有达成研究总目标。

## 服务端入口

SSH：ubuntu-server；项目：/home/login/data/cle/auto/GPT；Python：/home/login/miniconda3/envs/sunfuxiang/bin/python；Git：/home/login/miniconda3/bin/git。

test反馈目录：/home/login/data/cle/auto/GPT/experiments/ac_test_feedback_window01。

当前有效评估实现：run_test_feedback_v2.py、audit_test_feedback_v2.py和config_v2.json。最初v1在首次新推理之前因缺少psutil退出；v1源码、配置、日志和原因保留，v2改用已有/proc资源读取，没有安装或改动共享Python环境。旧v2监控循环在第一次完整39权重反馈之后，主动切换成coordinate_feedback.py；COORDINATOR_TRANSITION.json记录此操作，旧runner的SIGTERM不是未解决的模型故障。

coordinate_feedback.py只在新增完整方法组或主要baseline审计变化时触发推理，其余有效预测通过完整签名、checkpoint SHA、推理语义、清单和评估配置复用。coordinator_identity.json与/proc的实际命令用于判断活进程。原资格训练独立继续，不因后处理而重训。全部资格方案完成后应有15方法组/42个不同权重。

流水线结束后，可在相同环境重建反馈汇总和必要推理；完整有效预测会复用，runner.lock阻止同入口并发覆盖：

```bash
/home/login/miniconda3/envs/sunfuxiang/bin/python /home/login/data/cle/auto/GPT/experiments/ac_test_feedback_window01/run_test_feedback_v2.py
/home/login/miniconda3/envs/sunfuxiang/bin/python /home/login/data/cle/auto/GPT/experiments/ac_test_feedback_window01/summarize_test_feedback.py
```

原训练复现使用原冻结阶段的完整源码、配置、预训练SHA和命令，见第一、二阶段报告。禁止直接删掉complete后在原冻结目录重训；复现应在独立源码/实验/权重目录按原配置运行，避免覆盖历史。

## 注册权重的单图推理

predict_feedback_model.py调用原模型工厂并严格装载登记checkpoint，核验SHA，按A或B配方resize和中心裁剪。所有现有模型保留原3个类别ID，ID1为恶性；其余名称按原冻结数据映射解释。模型工厂仍需服务器上已登记的公开预训练文件，再覆盖完整训练权重。

```bash
/home/login/miniconda3/envs/sunfuxiang/bin/python /home/login/data/cle/auto/GPT/experiments/ac_test_feedback_window01/predict_feedback_model.py --method qualification_swin_t_B --seed 45 --image /absolute/path/example.png --output /home/login/data/cle/auto/GPT/experiments/ac_test_feedback_window01/cache/example_prediction.json
```

这是调用示例，不表示已将Swin-T384冻结为主要baseline，也不表示将最高seed作为最终模型。三seed完整比较仍以batch32的逐图预测及独立审计为准，单图AMP可能存在很小数值差异。

下一机制、忠实成熟方法参照、消融、门初始化、学习率预算、GPU数值检查及test反馈规则见NEXT_MECHANISM_PREREGISTRATION_ZH.md。持续研究每5分钟检查本窗口状态，普通运行和一般阴性保持安静；阶段完成后继续机制实施，不把快照视为总任务结束。
