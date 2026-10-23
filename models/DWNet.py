import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from .idynamic import IDynamicDWConv
from torch.autograd import Function
from collections import namedtuple
import cupy

Stream = namedtuple('Stream', ['ptr'])
from string import Template

def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_idynamic_kernel = kernel_loop + '''
extern "C"
__global__ void idynamic_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${top_height} / ${top_width};
    const int c = (index / ${top_height} / ${top_width}) % ${channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
            * ${bottom_width} + w_in;
          const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h)
            * ${top_width} + w;
          value += weight_data[offset_weight] * bottom_data[offset];
        }
      }
    }
    top_data[index] = value;
  }
}
'''


_idynamic_kernel_backward_grad_input = kernel_loop + '''
extern "C"
__global__ void idynamic_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
        const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
        if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
          const int h_out = h_out_s / ${stride_h};
          const int w_out = w_out_s / ${stride_w};
          if ((h_out >= 0) && (h_out < ${top_height})
                && (w_out >= 0) && (w_out < ${top_width})) {
            const int offset = ((n * ${channels} + c) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            value += weight_data[offset_weight] * top_diff[offset];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''


_idynamic_kernel_backward_grad_weight = kernel_loop + '''
extern "C"
__global__ void idynamic_backward_grad_weight_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* const buffer_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int kh = (index / ${kernel_w} / ${top_height} / ${top_width})
          % ${kernel_h};
    const int kw = (index / ${top_height} / ${top_width}) % ${kernel_w};
    const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
    const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
    if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
      const int g = (index / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${groups};
      const int n = (index / ${groups} / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${num};
      ${Dtype} value = 0;
      #pragma unroll
      for (int c = g * (${channels} / ${groups}); c < (g + 1) * (${channels} / ${groups}); ++c) {
        const int top_offset = ((n * ${channels} + c) * ${top_height} + h)
              * ${top_width} + w;
        const int bottom_offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
              * ${bottom_width} + w_in;
        value += top_diff[top_offset] * bottom_data[bottom_offset];
      }
      buffer_data[index] = value;
    } else {
      buffer_data[index] = 0;
    }
  }
}
'''


class _idynamic(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        assert input.dim() == 4 and input.is_cuda
        assert weight.dim() == 6 and weight.is_cuda
        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h = int((height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) / stride[0] + 1)
        output_w = int((width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) / stride[1] + 1)

        output = input.new(batch_size, channels, output_h, output_w)
        n = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel('idynamic_forward_kernel', _idynamic_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, channels=channels, groups=weight.size()[1],
                            bottom_height=height, bottom_width=width,
                            top_height=output_h, top_width=output_w,
                            kernel_h=kernel_h, kernel_w=kernel_w,
                            stride_h=stride[0], stride_w=stride[1],
                            dilation_h=dilation[0], dilation_w=dilation[1],
                            pad_h=padding[0], pad_w=padding[1])
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input, weight)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output.contiguous()
        input, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation

        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h, output_w = grad_output.size()[2:]

        grad_input, grad_weight = None, None

        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels, groups=weight.size()[1],
                   bottom_height=height, bottom_width=width,
                   top_height=output_h, top_width=output_w,
                   kernel_h=kernel_h, kernel_w=kernel_w,
                   stride_h=stride[0], stride_w=stride[1],
                   dilation_h=dilation[0], dilation_w=dilation[1],
                   pad_h=padding[0], pad_w=padding[1])

        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())

                n = grad_input.numel()
                opt['nthreads'] = n

                f = load_kernel('idynamic_backward_grad_input_kernel',
                                _idynamic_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())

                n = grad_weight.numel()
                opt['nthreads'] = n

                f = load_kernel('idynamic_backward_grad_weight_kernel',
                                _idynamic_kernel_backward_grad_weight, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad_input, grad_weight, None, None, None


def _idynamic_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """ idynamic kernel
    """
    assert input.size(0) == weight.size(0)
    assert input.size(-2) // stride == weight.size(-2)
    assert input.size(-1) // stride == weight.size(-1)
    if input.is_cuda:
        out = _idynamic.apply(input, weight, _pair(stride), _pair(padding), _pair(dilation))
        if bias is not None:
            out += bias.view(1, -1, 1, 1)
    else:
        raise NotImplementedError
    return out


class IDynamicDWConv(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size,
                 group_channels):
        super(IDynamicDWConv, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = group_channels
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.BatchNorm2d(channels // reduction_ratio),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels // reduction_ratio, kernel_size ** 2 * self.groups, 1)
        )

    def forward(self, x):
        weight = self.conv2(self.conv1(x))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        out = _idynamic_cuda(x, weight, stride=1, padding=(self.kernel_size - 1) // 2)
        return out

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicDWConv(nn.Module):
    def __init__(self, dim, kernel_size, bias=True, stride=1, padding=1, groups=1, reduction=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(dim, dim // reduction, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim // reduction, dim * kernel_size * kernel_size, 1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.conv2(self.relu(self.bn(self.conv1(self.pool(x)))))
        weight = weight.view(b * self.dim, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding,
                     groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x


class DWBlock(nn.Module):

    def __init__(self, dim, window_size, dynamic=False, inhomogeneous=False, heads=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.dynamic = dynamic
        self.inhomogeneous = inhomogeneous
        self.heads = heads

        # pw-linear
        self.conv0 = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(dim)

        if dynamic and not inhomogeneous:
            self.conv = DynamicDWConv(dim, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim)
        if dynamic and inhomogeneous:
            print(window_size, heads)
            self.conv = IDynamicDWConv(dim, window_size, heads)
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim)

        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

        # pw-linear
        self.conv2 = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, H, W, C = x.shape

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # x = self.conv0(x)
        flops += N * self.dim * self.dim
        # x = self.conv(x)
        if self.dynamic and not self.inhomogeneous:
            flops += (
                        N * self.dim + self.dim * self.dim / 4 + self.dim / 4 * self.dim * self.window_size * self.window_size)
        elif self.dynamic and self.inhomogeneous:
            flops += (
                        N * self.dim * self.dim / 4 + N * self.dim / 4 * self.dim / self.heads * self.window_size * self.window_size)
        flops += N * self.dim * self.window_size * self.window_size
        #  x = self.conv2(x)
        flops += N * self.dim * self.dim
        #  batchnorm + relu
        flops += 8 * self.dim * N
        return flops


class SpatialBlock(nn.Module):

    def __init__(self, dim, input_resolution, window_size=7,
                 mlp_ratio=4., drop=0., drop_path=0., dynamic=False, inhomogeneous=False, inhomo_head=None,
                 act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.dynamic = dynamic

        self.attn2conv = DWBlock(dim, window_size, dynamic, inhomogeneous, inhomo_head)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.attn2conv(x.view(B, H, W, C))
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        shortcut = x
        x = self.bn(x.view(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).view(B, H * W, C)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution} " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # batchnorm
        flops += 2 * self.dim * H * W
        # attn2conv
        flops += self.attn2conv.flops(H * W)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        return flops


class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, window_size,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, dynamic=False, inhomogeneous=False, inhomo_head=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SpatialBlock(dim=dim, input_resolution=input_resolution,
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         dynamic=dynamic,
                         inhomogeneous=inhomogeneous,
                         inhomo_head=inhomo_head,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class DWNet(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], window_size=7, mlp_ratio=4.,
                 drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 ape=False, patch_norm=True, use_checkpoint=False, dynamic=False, inhomogeneous=False,
                 inhomo_heads=None, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               dynamic=dynamic,
                               inhomogeneous=inhomogeneous,
                               inhomo_head=inhomo_heads[i_layer])

            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops