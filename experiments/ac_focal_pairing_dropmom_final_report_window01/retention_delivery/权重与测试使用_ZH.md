# 实际权重与测试方式

服务器项目根目录：/home/login/data/cle/auto/GPT（canonical /data/cle/auto/GPT）。Python：/home/login/miniconda3/envs/sunfuxiang/bin/python。

新入口严格按实际自定义结构加载权重，调用原evaluate.py字节SHA e5213828...中的compute_metrics、print_report、save_outputs等原函数。原evaluate.py不修改；其原主注册表/strictFalse过滤不适合该新结构。新入口默认原研究squareResize256/center224、AMP FP16 logits后FP32 softmax、物理B32/single；完整427val概率已对原记录误差0。

## 在服务器执行

```bash
cd /home/login/data/cle/auto/GPT
/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/ac_focal_pairing_dropmom_final_report_window01/delivery/evaluate_registered_network.py --arm mom_misaligned --seed 45 --split val --out_dir /home/login/data/cle/auto/GPT/experiments/ac_focal_pairing_dropmom_final_report_window01/manual_val_s45_run01
```

手动测试另一目录时，用 --data_path /你的绝对测试目录 替换 --split val；目录下按Benign_lesions、Malignant_lesions、body_of_stomach_normal三子目录放图。输出目录必须新建且尚不存在。要比较baseline，用同一入口 --arm plain，相同seed、目录、preprocess及precision。不要把另一个671图测试集结果与当前576图结果直接比较。

--seed 支持45/46/47；--arm 同时支持mom_only、misaligned_only作真实对照。--precision fp32可切换原evaluate主程序的FP32；--preprocess evaluate可切换原evaluate等比resize变换，必须对baseline一起同口径，不能替换原报告数字。--bootstrap调用原evaluate逐图1000 seed42区间，与本研究41患者20k配对条件区间不同。新入口默认完整427val模式已实际核验，其它数据/精度组合不被说成已验证提点。

保留原六位小数predictions.csv及原格式report/results，同时另写predictions_full_precision.csv和full_metrics_and_identity.json，保存未截断概率、全部指标和实际权重/源SHA。

## 原实际权重（均保留，不重训、不集成）

|seed|最早valbest epoch（从0计）|实际权重路径|SHA256|
|---|---:|---|---|
|45|34|weights/ac_midstage_focal_structure_window01/mfc_mom_misaligned_s45/model_best.pth|1fa9b8ee82d05ebf1ced3b341487c48b0655e8658ace75d939d1c27cdaf6d5fd|
|46|30|weights/ac_midstage_focal_structure_window01/mfc_mom_misaligned_s46/model_best.pth|0d10139d3b1df228635abb03bbd17a252f155754af0022c2282f3d81679a1d61|
|47|31|weights/ac_midstage_focal_structure_window01/mfc_mom_misaligned_s47/model_best.pth|3d79a205b9407b5cad2e0ed826a13688e67ecb2b1d4cb60374fa262d43aecc9f|

完整模型源码位于models/optimize/ac_midstage_focal_window01；新兼容库/纯去一控制位于models/optimize/ac_focal_pairing_dropmom_control_window01，LibraryNet支持原mom_misaligned。FocalNet受限参考与MIT原码/许可保留于原CPU原型references，原方向矩来源与完整历史证据不改。成熟算子组合不自动代表新颖。

完整主证据包SHA79cf3436a867cc3f906e483ee3d7e89794657a2a9a83f48d042207b64169dcb8；严格评估入口补包SHAce2e0f2804a64b9eb701f159dc28f77fd23ca3edeee318fdbffbcd7a27d0feae。本地完整和入口两个独立verifier均passed，参见完整轮/local_complete_verification.json及local_evaluation_entry_verification.json。
