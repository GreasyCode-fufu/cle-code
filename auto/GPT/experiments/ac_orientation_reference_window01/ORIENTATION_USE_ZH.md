# C4成熟参照的复现与使用

代码目录：/home/login/data/cle/auto/GPT/experiments/ac_orientation_reference_window01。SSH ubuntu-server，Python /home/login/miniconda3/envs/sunfuxiang/bin/python，Git /home/login/miniconda3/bin/git。

网络为已有Swin-T384，使用原seed45/46/47三个最早val0最佳checkpoint，路径及完整SHA在config.json/reference.json。成熟C4方法只将0/90/180/270度四个softmax概率等权平均，没有新训练和新网络参数。原公开预训练文件仍需留在资格阶段登记目录中，模型工厂随后严格覆盖完整训练状态。

论文/报告的96.5770%是三个模型分别评估后的macroF1平均值，不是最高seed，也不是三模型概率集成。固定主要B0为此前按val规则选出的ConvNeXt-Tiny224；其test均值94.9814%，C4平均差+1.5956百分点。仅固定集条件数值门槛通过，test参与开发且历史A–C构建用过其标签，不能称独立验证或结构创新完成。

每个seed的单图推理示例（seed仅用于明确复现对象，不用于挑最高分）：

```bash
/home/login/miniconda3/envs/sunfuxiang/bin/python /home/login/data/cle/auto/GPT/experiments/ac_orientation_reference_window01/predict_orientation_reference.py --seed 45 --image /absolute/path/example.png --output /home/login/data/cle/auto/GPT/experiments/ac_orientation_reference_window01/exports/example.json
```

输入RGB resize448双线性、中心crop384、原ImageNet归一化，然后四个90度整数旋转；不插值旋转。输出原三类ID及概率（ID1为恶性）。单图AMP形状与正式batch32不同，可有很小概率差异；完整研究比较以保存的全量逐图审计为准。

复算已有完整方法，按相同环境运行下列阶段。有效完成结果会按权重/预测SHA复用；其余角度和均值会独立复算指标，不重新训练：

```bash
/home/login/miniconda3/envs/sunfuxiang/bin/python /home/login/data/cle/auto/GPT/experiments/ac_orientation_reference_window01/run_orientation_reference.py analyze
/home/login/miniconda3/envs/sunfuxiang/bin/python /home/login/data/cle/auto/GPT/experiments/ac_orientation_reference_window01/audit_orientation_gate.py
```

完整推理阶段evaluate会复用签名匹配的complete，不盲目重跑；实际任务仍运行时先核对launch/bootstrap_status与/proc，不能靠锁文件或旧PID推断终止。模型完整概率、全部角度和三seed混淆矩阵见runs及orientation_results.json。

本方法每图4次forward、114.39149 GFLOPs，约原W0四倍计算；参数27,521,661不增加。不是一个廉价单forward新网络。新C4方向矩stem已在独立目录完成CPU数学/224和384预训练一致性与预算检查，增加18,528参数、一次forward约29.19241 GFLOPs，但尚未正式GPU预检和CLE训练，不能宣称已追平C4性能。结构研究继续。
