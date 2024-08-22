# RepViT: Revisiting Mobile CNN From ViT Perspective pytorch代码复现 进行cifar100图像分类
## Usage

python train.py 启动训练

## 复现流程

![image](F:\repvit-cifar10\repvit-cifar10\repvit.png)

基于这个整体架构进行设计

### 1 Conv_BN

根据repvit提出的结构重参数化，融合Conv和BN

```python
 class Conv_BN(nn.Module):
     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
    padding=0, dilation=1, groups=1, bn_weight_init=1):
         super().__init__()
         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
    padding, dilation, groups, bias=False)
         self.bn = nn.BatchNorm2d(out_channels)
         torch.nn.init.constant_(self.bn.weight, bn_weight_init)
         torch.nn.init.constant_(self.bn.bias, 0)
     def forward(self,x):
         return self.bn(self.conv(x))
```

### 2 FFN

如上图所示。为了实现单独且更深的下采样层，我们首先采用单个 1x1 卷积来调制通道维度，该卷积位于 深度卷积之后。这样一来，图中的两个1x1卷积的输入和输出就可以通过残差连接连 接起来，形成一个FFN

```python
class FFN(nn.Module):
     def __init__(self, in_channels, tmp_channels, out_channels, kernel_size=1, 
    stride=1, padding=0):
         super().__init__()
         assert in_channels == out_channels
         self.conv1 = Conv2d_BN(in_channels, tmp_channels, kernel_size)
         self.gelu = nn.GELU() #确保在移动端的兼容，使用GELU激活函数
         self.conv2 = Conv2d_BN(tmp_channels, out_channels, kernel_size, stride, 
        padding, bn_weight_init=0)
     def forward(self,x):
         return x+self.conv2(self.gelu(self.conv1(x))) #残差连接
```

### 3 更深的下采样层

```python
class DW(nn.Module):
     def __init__(self, dims):
         super().__init__()
         self.conv1 = Conv2d_BN(dims, dims, 3, 1, 1, groups=dims)
         self.conv2 = nn.Conv2d(dims, dims, 1, 1, 0, groups=dims)
         self.bn = nn.BatchNorm2d(dims)
         self.dims=dims
     def forward(self,x):
         return self.bn(x+self.conv1(x)+self.conv2(x))
```

### 4 简单的分类器

![image](F:\repvit-cifar10\repvit-cifar10\classifer.png)

如图b 相比于mobilenetv3，简化分类器设计，降低延迟

```python
class FC(nn.Module):
     def __init__(self, dims, num_classes):
         super().__init__()
         self.bn = nn.BatchNorm1d(dims)
         self.linear = nn.Linear(dims, num_classes, bias=True)
         trunc_normal_(self.linear.weight, std = 0.02)
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
```

### 5 RepVitBlock

块设计，针对总体架构图上的设计，该Block的应用分成stride为1和2两种情况讨论

```python
class RepViTBlock(nn.Module):
    def __init__(self, in_channels, tmp_channels, out_channels, kernel_size, 
stride, se):
        super().__init__()
        assert stride in [1,2]
        assert tmp_channels == 2 * in_channels
        self.flag = stride==1 and in_channels == out_channels
        if stride == 1 and self.flag:
            self.token_mixer = nn.Sequential(
                DW(in_channels),
                SqueezeExcite(in_channels,0.25) if se else nn.Identity(),
            )
            self.channel_mixer = FFN(in_channels, tmp_channels, out_channels)
        else:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(in_channels, in_channels, kernel_size, stride, 
(kernel_size-1)//2, groups=in_channels),
                SqueezeExcite(in_channels,0.25) if se else nn.Identity(),
                Conv2d_BN(in_channels, out_channels, kernel_size=1, stride=1, 
padding=0),
            )
            self.channel_mixer = FFN(out_channels, out_channels*2, out_channels)
    def forward(self,x):
        return self.channel_mixer(self.token_mixer(x))

```

### 6 RepVit

整合上述模块

```python
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
        for kerenl_size, expansion_ratio, channels_size, se, stride in 
self.configs:
            out_channels = _make_divisible(channels_size, 8)
            expanded_size = _make_divisible(in_channels * expansion_ratio, 8)
            layers.append(RepViTBlock(in_channels, expanded_size, out_channels, 
kerenl_size, stride, se))
            in_channels = out_channels
        self.features = nn.ModuleList(layers)
        self.classifier = Classfier(out_channels, num_classes)
#repvit0.9版本和1.5版本
def repvit_m0_9(pretrained=False, num_classes=100):
    configs = [
                [3, 2, 48, 1, 1],  # stage 1
                [3, 2, 48, 0, 1],  
                [3, 2, 48, 0, 1], 
                [3, 2, 96, 0, 2],  # 下采样
                [3, 2, 96, 1, 1],  # stage 2
                [3, 2, 96, 0, 1],  
                [3, 2, 96, 0, 1],  
                [3, 2, 192, 0, 2], #下采样
                [3, 2, 192, 1, 1], # stage3
                [3, 2, 192, 0, 1],
                [3, 2, 192, 1, 1], # stage3
                [3, 2, 192, 0, 1],
                [3, 2, 192, 1, 1], # stage3 
                [3, 2, 192, 0, 1],
                [3, 2, 192, 1, 1], # stage3
                [3, 2, 192, 0, 1],
                [3, 2, 192, 1, 1], # stage3
                [3, 2, 192, 0, 1],
                [3, 2, 192, 1, 1], # stage3
                [3, 2, 192, 0, 1],
                [3, 2, 192, 1, 1], # stage3
                [3, 2, 192, 0, 1],
                [3, 2, 192, 0, 1],
                [3, 2, 384, 0, 2], # 下采样
                [3, 2, 384, 1, 1], # stage 4
                [3, 2, 384, 0, 1]  
              ]
    return RepViT(configs, num_classes=num_classes)
 def repvit_m2_3(pretrained=False, num_classes = 100):
    """
    Constructs a MobileNetV3-Large model
    """
    configs = [
        # k, t, c, SE, s 
        [3,   2,  80, 1, 1],
        [3,   2,  80, 0, 1],
        [3,   2,  80, 1, 1],
        [3,   2,  80, 0, 1],
        [3,   2,  80, 1, 1],
        [3,   2,  80, 0, 1],
        [3,   2,  80, 0, 1],
        [3,   2,  160, 0, 2],
        [3,   2,  160, 1, 1],
        [3,   2,  160, 0, 1],
        [3,   2,  160, 1, 1],
        [3,   2,  160, 0, 1],
        [3,   2,  160, 1, 1],
        [3,   2,  160, 0, 1],
        [3,   2,  160, 0, 1],
        [3,   2,  320, 0, 2],
        [3,   2,  320, 1, 1],
        [3,   2,  320, 0, 1],
        [3,   2,  320, 1, 1],
        [3,   2,  320, 0, 1],
        [3,   2,  320, 1, 1],
        [3,   2,  320, 0, 1],
        [3,   2,  320, 1, 1],
        [3,   2, 320, 0, 1],
        [3,   2, 320, 1, 1],
        [3,   2, 320, 0, 1],
        [3,   2, 320, 1, 1],
        [3,   2, 320, 0, 1],
        [3,   2, 320, 1, 1],
        [3,   2, 320, 0, 1],
        [3,   2, 320, 1, 1],
        [3,   2, 320, 0, 1],
        [3,   2, 320, 1, 1],
        [3,   2, 320, 0, 1],
        [3,   2, 320, 1, 1],
        [3,   2, 320, 0, 1],
        [3,   2, 320, 1, 1],
        [3,   2, 320, 0, 1],
        [3,   2, 320, 1, 1],
        [3,   2, 320, 0, 1],
        [3,   2, 320, 1, 1],
        [3,   2, 320, 0, 1],
        [3,   2, 320, 1, 1],
        [3,   2, 320, 0, 1],
        [3,   2, 320, 1, 1],
        [3,   2, 320, 0, 1],
        [3,   2, 320, 1, 1],
        [3,   2, 320, 0, 1],
        [3,   2, 320, 1, 1],
        [3,   2, 320, 0, 1],
        # [3,   2, 320, 1, 1, 1],
        # [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 0, 1],
        [3,   2, 640, 0, 2],
        [3,   2, 640, 1, 1],
        [3,   2, 640, 0, 1],
        # [3,   2, 640, 1, 1, 1],
        # [3,   2, 640, 0, 1, 1]
    ]    
    return RepViT(configs, num_classes=num_classes) 	
```

## 复现结果

![image](F:\repvit-cifar10\repvit-cifar10\result.png)

mobilenetv3：71.92%

repvitm0.9   :   79.36%  

repvitm1.5   :   81.42%
