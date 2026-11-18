# 创建后的验证

日期：2026-11-09。此次仅验证新项目入口，不新增完整训练任务。

1. `bash run_reference.sh --check-only` 在sunfuxiang环境下成功，返回PASS。
2. 校验冻结核心源码、3080张图像的路径与内容、ImageNet预训练权重哈希全部通过。
3. seed42分类头初值哈希为5941f9195f990b3e6eacd29966896a7b803ecd32f5883cc164d25307c8bb8ee2，与成功复现环境一致。
4. 用cle_topnets直接执行同一check-only入口，返回非零并明确报告Python/PyTorch/Torchvision/NumPy/Pillow版本不匹配，符合预期。
5. 初始Git提交bbb29fe；项目归属login:login，旧项目未修改。

六次完整训练的证据来自本轮此前成功完成的historical_protocol_codex_20261109批次；复制了其日志、预测和复现审计作为reference，不把此次入口检查当作又完成了一批训练。
