import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class MambaLayer(nn.Module):
    def __init__(self, in_chs=128, dim=128, d_state=16, d_conv=4, expand=2, last_feat_size=16):
        super().__init__()

        # 根据不同的最后特征尺寸生成池化尺度
        pool_scales = self.generate_arithmetic_sequence(1, last_feat_size, last_feat_size // 4)
        self.pool_len = len(pool_scales)

        # 构建池化层
        self.pool_layers = nn.ModuleList()
        self.pool_layers.append(nn.Sequential(
            ConvBNReLU(in_chs, dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)  # 直接池化成一个1x1的特征
        ))

        # 对于其它尺度，执行不同尺度的池化
        for pool_scale in pool_scales[1:]:
            self.pool_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),  # 在不同尺度上进行池化
                    ConvBNReLU(in_chs, dim, kernel_size=1)
                ))
 
        # Mamba 模型实例
        self.mamba = Mamba(
            d_model=dim * self.pool_len + in_chs,  # 模型维度 d_model
            d_state=d_state,  # SSM 状态扩展因子
            d_conv=d_conv,  # 局部卷积宽度
            expand=expand  # 扩展因子
        )

    def forward(self, x):
        res = x
        B, C, H, W = res.shape
        ppm_out = [res]

        # 执行不同尺度的池化操作
        for p in self.pool_layers:
            pool_out = p(x)
            pool_out = F.interpolate(pool_out, (H, W), mode='bilinear', align_corners=False)  # 将池化结果上采样回原始尺寸
            ppm_out.append(pool_out)

        # 将多尺度的池化结果拼接在一起
        x = torch.cat(ppm_out, dim=1)
        _, chs, _, _ = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c', b=B, c=chs, h=H, w=W)

        # 使用 Mamba 模型处理
        x = self.mamba(x)
        x = x.transpose(2, 1).view(B, chs, H, W)
        return x

    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence

if __name__ == '__main__':
    # 假设输入图像大小为 [B, 512, 256, 256]，通道数为512，尺寸为256x256
    x = torch.randn(8, 128, 256, 256)  # 输入的Tensor

    # 确保你的设备支持CUDA，并且将模型和数据移动到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mamba_layer = MambaLayer(in_chs=128, dim=32, d_state=16, d_conv=4, expand=2, last_feat_size=16).to(device)
    x = x.to(device)  # 将输入数据移动到GPU上

    output = mamba_layer(x)
    print(output.shape)  # 输出的Tensor大小
