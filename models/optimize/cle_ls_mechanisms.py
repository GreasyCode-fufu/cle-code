"""Controlled LS kernel-law diagnostics; mature reference, no novelty claim."""
from torch import nn
from models.optimize.cle_morphology import KernelPredictor, PlainControl, shared_aggregate


class KernelLawLS(nn.Module):
    def __init__(self, channels, law):
        super().__init__()
        self.predictor = KernelPredictor(channels)
        self.norm = nn.BatchNorm2d(channels)
        self.law = law

    def weights(self, x):
        value = self.predictor(x, positive=self.law == 'positive')
        if self.law == 'global':
            value = value.mean((-2, -1), keepdim=True).expand_as(value)
        return value

    def forward(self, x):
        return self.norm(shared_aggregate(x, self.weights(x)))


class LawResidual(nn.Module):
    def __init__(self, branch=None):
        super().__init__()
        self.branch = branch
        self.enabled = True

    def forward(self, x):
        return x + .1 * self.branch(x) if self.enabled and self.branch is not None else x


class LSStudyResNet(nn.Module):
    def __init__(self, backbone, arm, control_width):
        super().__init__()
        self.backbone = backbone
        branch = None
        if arm == 'conv_ls_budget':
            branch = PlainControl(256, control_width)
        elif arm.startswith('ls_'):
            branch = KernelLawLS(256, {'ls_signed': 'signed', 'ls_positive': 'positive', 'ls_global': 'global'}[arm])
        elif arm != 'plain':
            raise ValueError(arm)
        self.modifier = LawResidual(branch)

    def forward(self, x):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer2(b.layer1(x))
        x = self.modifier(b.layer3(x))
        x = b.layer4(x)
        return b.fc(b.avgpool(x).flatten(1))
