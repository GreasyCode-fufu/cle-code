"""CLE EGB/SCG transfer into MobileNet-V3-Large depthwise convolutions."""
import torch
from torch import nn
from models.MobileNetV3 import mobilenet_v3_large
from models.optimize.convnext_cle2 import _PFECore, StripBranch


class CLEDepthwise(nn.Conv2d):
    def __init__(self, original, use_egb, use_scg):
        super().__init__(original.in_channels, original.out_channels,
                         original.kernel_size, original.stride, original.padding,
                         original.dilation, original.groups, original.bias is not None,
                         original.padding_mode)
        self.load_state_dict(original.state_dict())
        self.use_egb, self.use_scg = use_egb, use_scg
        if use_egb:
            self.egb = _PFECore(base_ratio=0.1)
            self.egb_g = nn.Parameter(torch.zeros(1))
        if use_scg:
            self.scg = StripBranch(original.out_channels, k2=19)

    def forward(self, x):
        x = super().forward(x)
        if self.use_egb:
            # FFT remains float32; preserve the backbone activation dtype for AMP.
            x = x + self.egb_g.to(x.dtype) * self.egb(x).to(x.dtype) * x
        if self.use_scg:
            x = x + self.scg(x)
        return x


def mobilenet_v3_large_cle(num_classes=3):
    model = mobilenet_v3_large(num_classes=num_classes)
    # Protect the common classifier initialization and subsequent DataLoader RNG.
    with torch.random.fork_rng(devices=[]):
        for i in range(4, 16):
            block = model.features[i].block
            found = []
            for j, layer in enumerate(block):
                if isinstance(layer, nn.Sequential) and isinstance(layer[0], nn.Conv2d):
                    conv = layer[0]
                    if conv.groups == conv.in_channels and conv.kernel_size != (1, 1):
                        layer[0] = CLEDepthwise(conv, use_egb=True, use_scg=i >= 7)
                        found.append(j)
            assert len(found) == 1, (i, found)
    return model
