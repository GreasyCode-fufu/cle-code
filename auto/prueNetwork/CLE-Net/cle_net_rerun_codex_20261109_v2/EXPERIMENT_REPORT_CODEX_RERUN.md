# CLE_ac Codex独立复跑汇总

输出目录：/home/login/data/cle/auto/prueNetwork/CLE-Net/cle_net_rerun_codex_20261109_v2

本次由Codex独立启动，使用新目录；没有读取 weights_r6b_recheck 作为CLE-Net或baseline复跑结果。每个模型seed42/43/44顺序训练并测试。

模型 | Macro-F1 | Accuracy | Macro-AUROC
---|---:|---:|---:
CLE-Net (ConvNeXt-T + EGB + SCG) | 0.9510±0.0056 | 0.9508±0.0056 | 0.9942±0.0010
ConvNeXt-T baseline | 0.9547±0.0029 | 0.9549±0.0030 | 0.9961±0.0010
MambaOut-Tiny | 0.9572±0.0117 | 0.9572±0.0116 | 0.9953±0.0019
TransNeXt-Tiny | 0.9523±0.0118 | 0.9525±0.0116 | 0.9946±0.0016
MambaVision-T | 0.9449±0.0156 | 0.9456±0.0156 | 0.9936±0.0017
ThinkingViT-DeiT 3H->6H | 0.9395±0.0194 | 0.9398±0.0201 | 0.9924±0.0022
CPUBone-B2 | 0.9189±0.0137 | 0.9207±0.0123 | 0.9894±0.0028

复跑manifest：LAUNCH_MANIFEST.txt；六个测试结果均n_total=576。