import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['StableADownRes']


# =========================
# Conv
# =========================

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [
            d * (x - 1) + 1 for x in k
        ]

    if p is None:
        p = k // 2 if isinstance(k, int) else [
            x // 2 for x in k
        ]

    return p



class Conv(nn.Module):

    default_act = nn.SiLU()

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k,p,d),
            groups=g,
            dilation=d,
            bias=False
        )

        self.bn = nn.BatchNorm2d(c2)

        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )


    def forward(self,x):

        return self.act(
            self.bn(
                self.conv(x)
            )
        )



# =================================================
# Stable Feature Calibration
# =================================================

class StableFeatureCalibration(nn.Module):
    """
    Stable representation inspired feature calibration.

    Motivation:
    Learn task-relevant stable responses while
    suppressing unstable feature variations.
    """

    def __init__(
        self,
        channels,
        reduction=16
    ):
        super().__init__()


        hidden = max(
            channels // reduction,
            4
        )


        self.pool = nn.AdaptiveAvgPool2d(1)


        self.fc = nn.Sequential(

            nn.Conv2d(
                channels,
                hidden,
                1,
                bias=False
            ),

            nn.BatchNorm2d(hidden),

            nn.SiLU(),


            nn.Conv2d(
                hidden,
                channels,
                1,
                bias=False
            ),

            nn.Sigmoid()

        )


    def forward(self,x):

        weight = self.fc(
            self.pool(x)
        )


        return x * weight



# =================================================
# Triplet Attention
# =================================================


class BasicConv(nn.Module):

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0
    ):

        super().__init__()


        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size,
            stride,
            padding,
            bias=False
        )


        self.bn = nn.BatchNorm2d(
            out_planes
        )


        self.relu = nn.ReLU()



    def forward(self,x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x



class ZPool(nn.Module):

    def forward(self,x):

        return torch.cat(
            [
                torch.max(
                    x,
                    dim=1,
                    keepdim=True
                )[0],

                torch.mean(
                    x,
                    dim=1,
                    keepdim=True
                )
            ],
            dim=1
        )



class AttentionGate(nn.Module):

    def __init__(self):

        super().__init__()

        self.compress = ZPool()

        self.conv = BasicConv(
            2,
            1,
            kernel_size=7,
            padding=3
        )



    def forward(self,x):

        scale = torch.sigmoid(
            self.conv(
                self.compress(x)
            )
        )

        return x * scale



class TripletAttention(nn.Module):

    def __init__(
        self,
        no_spatial=False
    ):

        super().__init__()


        self.cw = AttentionGate()

        self.hc = AttentionGate()


        self.no_spatial = no_spatial


        if not no_spatial:

            self.hw = AttentionGate()



    def forward(self,x):


        # channel-width
        x_perm1 = x.permute(
            0,2,1,3
        ).contiguous()


        x_out1 = self.cw(
            x_perm1
        )


        x_out11 = x_out1.permute(
            0,2,1,3
        ).contiguous()



        # height-channel

        x_perm2 = x.permute(
            0,3,2,1
        ).contiguous()


        x_out2 = self.hc(
            x_perm2
        )


        x_out21 = x_out2.permute(
            0,3,2,1
        ).contiguous()



        if not self.no_spatial:

            x_out = self.hw(x)

            return (
                x_out +
                x_out11 +
                x_out21
            ) / 3


        else:

            return (
                x_out11 +
                x_out21
            ) / 2



# =================================================
# Stable Adaptive Downsampling Residual
# =================================================


class StableADownRes(nn.Module):


    def __init__(
        self,
        c1,
        c2,
        reduction=16
    ):

        super().__init__()



        # stable representation calibration

        self.stable = StableFeatureCalibration(
            c1,
            reduction
        )


        # triplet attention

        self.attention = TripletAttention()



        self.c = c2 // 2



        # learnable downsampling branch

        self.cv1 = Conv(
            c1//2,
            self.c,
            k=3,
            s=2,
            p=1
        )



        # semantic compression branch

        self.cv2 = Conv(
            c1//2,
            self.c,
            k=1,
            s=1
        )




    def forward(self,x):


        # 1. stable representation calibration

        x = self.stable(x)



        # 2. spatial/channel attention

        x = self.attention(x)



        # 3. split features

        x = F.avg_pool2d(
            x,
            kernel_size=2,
            stride=1,
            padding=0
        )


        x1,x2 = x.chunk(
            2,
            dim=1
        )


        # branch 1:
        # learnable downsampling

        x1 = self.cv1(x1)



        # branch 2:
        # structure preserving

        x2 = F.max_pool2d(
            x2,
            kernel_size=3,
            stride=2,
            padding=1
        )


        x2 = self.cv2(x2)



        return torch.cat(
            [
                x1,
                x2
            ],
            dim=1
        )



# ===========================
# test
# ===========================

if __name__ == "__main__":


    x = torch.randn(
        1,
        32,
        128,
        128
    )


    model = StableADownRes(
        32,
        64
    )


    y = model(x)


    print(
        "Output:",
        y.shape
    )