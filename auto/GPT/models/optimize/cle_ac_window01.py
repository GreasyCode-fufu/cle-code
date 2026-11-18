"""CLE cross-depth neighborhood readout: a falsifiable experiment, no novelty claim."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class NeighborhoodReadout(nn.Module):
    def __init__(self, variant, rank=96):
        super().__init__()
        self.variant = variant
        self.down = nn.Linear(384, rank)
        self.norm = nn.LayerNorm(rank)
        self.depth_logits = nn.Parameter(torch.zeros(3))
        self.up = nn.Linear(2 * rank, 768)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.enabled = True

    def forward(self, layers):
        if self.variant in ('no_depth', 'final_only'):
            layers = [layers[-1]] * 3
        statistics = []
        for tokens in layers:
            z = self.norm(F.gelu(self.down(tokens[:, 1:])))
            b, n, c = z.shape
            side = int(n ** .5)
            assert side * side == n
            if self.variant in ('no_neighbor', 'final_only'):
                neighbor = z
            else:
                grid = z.transpose(1, 2).reshape(b, c, side, side)
                neighbor = F.avg_pool2d(grid, 3, 1, 1, count_include_pad=False).flatten(2).transpose(1, 2)
            statistics.append(torch.cat((z.mean(1), (z * neighbor).mean(1)), -1))
        pooled = (torch.stack(statistics, 1) * self.depth_logits.softmax(0)[None, :, None]).sum(1)
        return self.up(pooled)


class CLEDepthNet(nn.Module):
    def __init__(self, root, variant):
        super().__init__()
        from pretrained_representations import dino
        self.variant = variant
        self.encoder = dino(root).requires_grad_(True)
        # Match the historical DINO baseline RNG order before adding modules.
        self.head = nn.Linear(768, 3)
        self.captures = {}
        self.readout = None
        if variant in ('relational', 'no_neighbor', 'no_depth', 'final_only'):
            self.readout = NeighborhoodReadout(variant)
        elif variant == 'official4':
            self.head4 = nn.Linear(1920, 3)
            # Keep unused historical head out of optimizer and parameter counts.
            del self.head
        elif variant != 'baseline':
            raise ValueError(variant)
        indices = [8, 9, 10, 11] if variant == 'official4' else [3, 7, 11]
        for index in indices:
            self.encoder.blocks[index].register_forward_hook(self._capture(index))

    def _capture(self, index):
        def callback(module, inputs, output):
            self.captures[index] = output
        return callback

    def forward(self, image):
        self.captures.clear()
        tokens = self.encoder.forward_features(image)
        base = torch.cat((tokens[:, 0], tokens[:, 1:].mean(1)), -1)
        if self.variant == 'official4':
            layers = [self.encoder.norm(self.captures[k]) for k in [8, 9, 10, 11]]
            result = self.head4(torch.cat([t[:, 0] for t in layers] + [tokens[:, 1:].mean(1)], -1))
        else:
            if self.readout is not None and self.readout.enabled:
                layers = [self.encoder.norm(self.captures[k]) for k in [3, 7, 11]]
                base = base + self.readout(layers)
            result = self.head(base)
        self.captures.clear()
        return result


class ConvNeXtReference(nn.Module):
    def __init__(self, root):
        super().__init__()
        from torchvision.models import convnext_base
        raw = torch.load(Path('/home/login/.cache/torch/hub/checkpoints/convnext_base-6075fbad.pth'), map_location='cpu', weights_only=True)
        model = convnext_base(weights=None, stochastic_depth_prob=.1)
        model.load_state_dict(raw, strict=True)
        self.encoder = nn.Sequential(model.features, model.avgpool, model.classifier[0], nn.Flatten(1))
        self.head = nn.Linear(1024, 3)
        self.variant = 'convnext'

    def forward(self, image):
        return self.head(self.encoder(image))
