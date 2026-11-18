"""R6 对比实验: 近年轻量主干(timm 实现 + 官方 ImageNet-1k 权重, 2026-11-09)。
权重: HF timm/<tag>/model.safetensors 下载后转存为 state_dict(pth/timm_r6/*.pth, 无 pickle 来源);
严格加载走 train_cle_fair.load_pretrained_weights_strict(仅 head/fc/classifier 根允许缺失)。
- mobilenetv4_conv_small: MobileNetV4-Conv-S (Qin et al., ECCV 2024), tag e2400_r224_in1k
- starnet_s2: StarNet-S2 (Ma et al., "Rewrite the Stars", CVPR 2024), tag in1k
"""
import timm


def mobilenetv4_conv_small(num_classes=3):
    return timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=num_classes)


def starnet_s2(num_classes=3):
    return timm.create_model("starnet_s2", pretrained=False, num_classes=num_classes)
