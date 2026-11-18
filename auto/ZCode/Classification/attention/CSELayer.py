import torch.nn as nn
import torch

class CSELayer(nn.Module):
    def __init__(self,in_channel, channel, reduction = 16):
        super(CSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Sequential(
                        nn.Linear(channel, channel // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel // reduction, channel),
                        nn.Sigmoid()
                )
        if in_channel != channel:
            self.att_fc = nn.Sequential(
                nn.Linear(in_channel, channel),
                nn.LayerNorm(channel),
                nn.ReLU(inplace=True)
            )
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.LayerNorm(channel),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        b, c, _, _ = x[0].size()
        gap = self.avg_pool(x[0]).view(b, c)
        if x[1] is None:
            all_att = self.fc(gap)
        else:
            pre_att = self.att_fc(x[1]) if hasattr(self, 'att_fc') else x[1]
            all_att = torch.cat((gap.view(b, 1, 1, c), pre_att.view(b, 1, 1, c)), dim=1)
            all_att = self.conv(all_att).view(b, c)
            all_att = self.fc(all_att)
        return {0: x[0] * all_att.view(b, c, 1, 1), 1: gap*all_att}
