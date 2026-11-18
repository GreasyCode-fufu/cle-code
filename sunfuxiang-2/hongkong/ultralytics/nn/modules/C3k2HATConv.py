# C3k2.py  
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  


def autopad(k, p=None):  
    """Calculate padding."""  
    p = p if p is not None else k // 2  
    return p 

  
class Conv(nn.Module):  
    """Convolution module with optional batchnorm and activation."""  
  
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  
        """Initialize a convolution layer with optional batchnorm and activation."""  
        super().__init__()  
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  
        self.bn = nn.BatchNorm2d(c2) if c2 != 1 else nn.Identity()  
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())  
  
    def forward(self, x):  
        """Forward pass of the convolution module."""  
        return self.act(self.bn(self.conv(x)))    

class HetConv(nn.Module):
 
    def __init__(self, input_channels, output_channels, stride=1, p=4):
        """
        Initialize the HetConv class.
        :param input_channels: the number of input channels
        :param output_channels: the number of output channels
        :param stride: convolution stride
        :param p: the value of P used in HetConv
        """
        super(HetConv, self).__init__()
        self.p = p
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filters = nn.ModuleList()
        self.convolution_1x1_index = []
        # Compute the indices of input channels fed to 1x1 convolutional kernels in all filters.
        # These indices of input channels are also the indices of the 1x1 convolutional kernels in the filters.
        # This is only executed when the HetConv class is created,
        # and the execution time is not included during inference.
        for i in range(self.p):
            self.convolution_1x1_index.append(self.compute_convolution_1x1_index(i))
        # Build HetConv filters.
        for i in range(self.p):
            self.filters.append(self.build_HetConv_filters(stride, p))
 
    def compute_convolution_1x1_index(self, i):
        """
        Compute the indices of input channels fed to 1x1 convolutional kernels in the i-th branch of filters (i=0, 1, 2,…, P-1). The i-th branch of filters consists of the {i, i+P, i+2P,…, i+N-P}-th filters.
        :param i: the i-th branch of filters in HetConv
        :return: return the required indices of input channels
        """
        index = [j for j in range(0, self.input_channels)]
        # Remove the indices of input channels fed to 3x3 convolutional kernels in the i-th branch of filters.
        while i < self.input_channels:
            index.remove(i)
            i += self.p
        return index
 
    def build_HetConv_filters(self, stride, p):
        """
        Build N/P filters in HetConv.
        :param stride: convolution stride
        :param p: the value of P used in HetConv
        :return: return N/P HetConv filters
        """
        temp_filters = nn.ModuleList()
        # nn.Conv2d arguments: nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        temp_filters.append(nn.Conv2d(self.input_channels//p, self.output_channels//p, 3, stride, 1, bias=False))
        temp_filters.append(nn.Conv2d(self.input_channels-self.input_channels//p, self.output_channels//p, 1, stride, 0, bias=False))
        return temp_filters
 
    def forward(self, input_data):
        """
        Define how HetConv processes the input images or input feature maps.
        :param input_data: input images or input feature maps
        :return: return output feature maps
        """
        output_feature_maps = []
        # Loop P times to get output feature maps. The number of output feature maps = the batch size.
        for i in range(0, self.p):
 
            # M/P HetConv filter kernels perform the 3x3 convolution and output to N/P output channels.
            output_feature_3x3 = self.filters[i][0](input_data[:, i::self.p, :, :])
            # (M-M/P) HetConv filter kernels perform the 1x1 convolution and output to N/P output channels.
            output_feature_1x1 = self.filters[i][1](input_data[:, self.convolution_1x1_index[i], :, :])
 
            # Obtain N/P output feature map channels.
            output_feature_map = output_feature_1x1 + output_feature_3x3
 
            # Append N/P output feature map channels.
            output_feature_maps.append(output_feature_map)
 
        # Get the batch size, number of output channels (N/P), height and width of output feature map.
        N, C, H, W = output_feature_maps[0].size()
        # Change the value of C to the number of output feature map channels (N).
        C = self.p * C
        # Arrange the output feature map channels to make them fit into the shifted manner.
        return torch.cat(output_feature_maps, 1).view(N, self.p, C//self.p, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

    

class C3k(nn.Module):  
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""  
  
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):  
        """Initializes the C3k module with specified channels, number of layers, and configurations."""  
        super().__init__()  
        c_ = int(c2 * e)  # hidden channels  
        self.cv1 = Conv(c1, c_, k, 1)  
        self.cv2 = Conv(c_, c2, k, 1, g=g)  
        self.add = shortcut and c1 == c2  
  
    def forward(self, x):  
        """Forward pass through the C3k module."""  
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))  
  

 
 
class Bottleneck(nn.Module):
    """Standard bottleneck with HetConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, p=4):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # 使用 HetConv 替换 Conv
        self.cv1 = HetConv(c1, c_, stride=1, p=p)
        self.cv2 = HetConv(c_, c2, stride=1, p=p)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data using HetConv."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k2HATConv(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, using HetConv in Bottleneck."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, p=4):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, self.c, 1, 1)  # 这里保持 Conv 不变
        self.cv2 = Conv(c1, self.c, 1, 1)  # 这里保持 Conv 不变
        self.cv3 = Conv((2 + n) * self.c, c2, 1)  # 这里保持 Conv 不变

        # 修改 m 列表中的模块为使用 HetConv 的 Bottleneck 或 C3k
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g, k=3) if c3k else
            Bottleneck(self.c, self.c, shortcut, g, p=p) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C3k2 layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))
 
 
 

 
 
