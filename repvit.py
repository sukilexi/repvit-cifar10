import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite
from timm.models.vision_transformer import trunc_normal_


def _make_divisible(value, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class Conv2d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        return self.bn(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        bn_gamma = self.bn.weight
        bn_beta = self.bn.bias
        mean = self.bn.running_mean
        var = self.bn.running_var
        std = torch.sqrt(var + self.bn.eps)
        fused_weight = self.conv.weight * (bn_gamma / std)[:, None, None, None]
        fused_bias = bn_beta - mean / std * bn_gamma
        fused_conv = nn.Conv2d(in_channels=fused_weight.size(1) * self.conv.groups,
                               out_channels=fused_weight.size(0),
                               kernel_size=fused_weight.shape[2:],
                               stride=self.conv.stride,
                               padding=self.conv.padding,
                               dilation=self.conv.dilation,
                               groups=self.conv.groups,
                               device=self.conv.weight.device)
        fused_conv.weight.data.copy_(fused_weight)
        fused_conv.bias.data.copy_(fused_bias)
        return fused_conv


class FFN(nn.Module):
    def __init__(self, in_channels, tmp_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        assert in_channels == out_channels
        self.conv1 = Conv2d_BN(in_channels, tmp_channels, kernel_size)
        self.gelu = nn.GELU()
        self.conv2 = Conv2d_BN(tmp_channels, out_channels, kernel_size, stride, padding, bn_weight_init=0)

    def forward(self, x):
        return x + self.conv2(self.gelu(self.conv1(x)))


class DW(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv1 = Conv2d_BN(dims, dims, 3, 1, 1, groups=dims)
        self.conv2 = Conv2d_BN(dims, dims, 1, 1, 0, groups=dims)
        self.bn = nn.BatchNorm2d(dims)
        self.dims = dims

    def forward(self, x):
        return self.bn(x) + self.conv1(x) + self.conv2(x)

    @torch.no_grad()
    def fuse(self):
        conv1 = self.conv1.fuse()
        conv2 = self.conv2
        bn = self.bn
        conv2.weight = torch.nn.functional.pad(conv2.weight, [1, 1, 1, 1])
        identity = torch.nn.functional.pad(
            torch.ones(conv2.weight.shape[0], conv2.weight.shape[1], 1, 1, device=conv2.weight.device), [1, 1, 1, 1])
        fused_conv = conv1
        fused_conv.weight.data.copy_(conv1.weight + conv2.weight + identity)
        fused_conv.bias.data.copy_(conv1.bias + conv2.bias)
        weight = bn.weight / (bn.running_var + bn.eps) ** 0.5
        weight = fused_conv.weight * weight[:, None, None, None]
        bias = bn.bias + (fused_conv.bias - bn.running_mean) * bn.weight / \
               (bn.running_var + bn.eps) ** 0.5
        fused_conv.weight.data.copy_(weight)
        fused_conv.bias.data.copy_(bias)
        return fused_conv


class RepViTBlock(nn.Module):
    def __init__(self, in_channels, tmp_channels, out_channels, kernel_size, stride, se):
        super().__init__()
        assert stride in [1, 2]
        assert tmp_channels == 2 * in_channels
        self.flag = stride == 1 and in_channels == out_channels
        if stride == 1 and self.flag:
            self.token_mixer = nn.Sequential(
                DW(in_channels),
                SqueezeExcite(in_channels, 0.25) if se else nn.Identity(),
            )
            self.channel_mixer = FFN(in_channels, tmp_channels, out_channels)
        else:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(in_channels, in_channels, kernel_size, stride, (kernel_size - 1) // 2, groups=in_channels),
                SqueezeExcite(in_channels, 0.25) if se else nn.Identity(),
                Conv2d_BN(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            )
            self.channel_mixer = FFN(out_channels, out_channels * 2, out_channels)

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class FC(nn.Module):
    def __init__(self, dims, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(dims)
        self.linear = nn.Linear(dims, num_classes, bias=True)
        trunc_normal_(self.linear.weight, std=0.02)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def foward(self, x):
        return self.linear(self.bn(x))


class Classfier(nn.Module):
    def __init__(self, dims, num_classes):
        super().__init__()
        self.fc = FC(dims, num_classes)

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class RepViT(nn.Module):
    def __init__(self, configs, num_classes=100):
        super().__init__()
        self.configs = configs
        in_channels = self.configs[0][2]

        stem = nn.Sequential(Conv2d_BN(3, in_channels // 2, 3, 2, 1),
                             nn.GELU(),
                             Conv2d_BN(in_channels // 2, in_channels, 3, 2, 1)
                             )
        layers = [stem]

        for kerenl_size, expansion_ratio, channels_size, se, stride in self.configs:
            out_channels = _make_divisible(channels_size, 8)
            expanded_size = _make_divisible(in_channels * expansion_ratio, 8)
            layers.append(RepViTBlock(in_channels, expanded_size, out_channels, kerenl_size, stride, se))
            in_channels = out_channels
        self.features = nn.ModuleList(layers)
        self.classifier = Classfier(out_channels, num_classes)

    def forward(self, x):
        for f in self.features:
            x = f(x)
        return self.classifier(x)


def repvit_m0_9(pretrained=False, num_classes=100):
    configs = [
        [3, 2, 48, 1, 1],  # stage 1
        [3, 2, 48, 0, 1],
        [3, 2, 48, 0, 1],
        [3, 2, 96, 0, 2],  # 下采样
        [3, 2, 96, 1, 1],  # stage 2
        [3, 2, 96, 0, 1],
        [3, 2, 96, 0, 1],
        [3, 2, 192, 0, 2],  # 下采样
        [3, 2, 192, 1, 1],  # stage3
        [3, 2, 192, 0, 1],
        [3, 2, 192, 1, 1],  # stage3
        [3, 2, 192, 0, 1],
        [3, 2, 192, 1, 1],  # stage3
        [3, 2, 192, 0, 1],
        [3, 2, 192, 1, 1],  # stage3
        [3, 2, 192, 0, 1],
        [3, 2, 192, 1, 1],  # stage3
        [3, 2, 192, 0, 1],
        [3, 2, 192, 1, 1],  # stage3
        [3, 2, 192, 0, 1],
        [3, 2, 192, 1, 1],  # stage3
        [3, 2, 192, 0, 1],
        [3, 2, 192, 0, 1],
        [3, 2, 384, 0, 2],  # 下采样
        [3, 2, 384, 1, 1],  # stage 4
        [3, 2, 384, 0, 1]
    ]
    return RepViT(configs, num_classes=num_classes)


def repvit_m2_3(pretrained=False, num_classes=100):
    """
    Constructs a MobileNetV3-Large model
    """
    configs = [
        # k, t, c, SE, s
        [3, 2, 80, 1, 1],
        [3, 2, 80, 0, 1],
        [3, 2, 80, 1, 1],
        [3, 2, 80, 0, 1],
        [3, 2, 80, 1, 1],
        [3, 2, 80, 0, 1],
        [3, 2, 80, 0, 1],
        [3, 2, 160, 0, 2],
        [3, 2, 160, 1, 1],
        [3, 2, 160, 0, 1],
        [3, 2, 160, 1, 1],
        [3, 2, 160, 0, 1],
        [3, 2, 160, 1, 1],
        [3, 2, 160, 0, 1],
        [3, 2, 160, 0, 1],
        [3, 2, 320, 0, 2],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 320, 1, 1],
        [3, 2, 320, 0, 1],
        # [3,   2, 320, 1, 1, 1],
        # [3,   2, 320, 0, 1, 1],
        [3, 2, 320, 0, 1],
        [3, 2, 640, 0, 2],
        [3, 2, 640, 1, 1],
        [3, 2, 640, 0, 1],
        # [3,   2, 640, 1, 1, 1],
        # [3,   2, 640, 0, 1, 1]
    ]
    return RepViT(configs, num_classes=num_classes)




