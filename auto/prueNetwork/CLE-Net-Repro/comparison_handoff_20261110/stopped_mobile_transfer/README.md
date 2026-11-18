# MobileNet-V3-Large + CLE EGB/SCG transfer

Independent Codex experiment on CLE_ac, seeds 42/43/44.
EGB: features 4–15; SCG: features 7–15, inside depthwise conv before BN/activation. Original SE retained.
Zero-gated EGB and zero-final-projection SCG preserve initial backbone function. Common RNG preserved during adapter creation.
Uses frozen CLE-Net-Repro training/evaluation code, dataset split and ImageNet normalization. Both classifier Linear layers are reinitialized as in the historical loader. No test-driven tuning.
300 maximum epochs, validation Macro-F1 selects checkpoint, patience 30. Baseline is freshly retrained, not borrowed from Zcode.
Sequential runs wait for an empty GPU before each train; no existing process is terminated.
This is an architectural transfer, not the original ConvNeXt CLE-Net. No guaranteed improvement.
See protocol.json, preflight.json, run_meta.json, logs and results for evidence.
