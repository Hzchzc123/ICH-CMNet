import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设输入图像的大小是 [B, 3, H, W]
class Channel_adapt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Channel_adapt, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)


class Verse_Channel_adapt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Verse_Channel_adapt, self).__init__()
        self.conv = nn.ConvTranspose2d(
    in_channels=in_channels,    # 输入通道数
    out_channels=out_channels,  # 输出通道数
    kernel_size=3,    # 卷积核大小
    stride=1,         # 步幅
    padding=1         # 填充
)

    def forward(self, x):
        return self.conv(x)

class Verse_adapt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Verse_adapt, self).__init__()
        self.conv = nn.ConvTranspose2d(
    in_channels=in_channels,    # 输入通道数
    out_channels=out_channels,  # 输出通道数
    kernel_size=4,    # 卷积核大小
    stride=2,         # 步幅
    padding=1         # 填充
)

    def forward(self, x):
        return self.conv(x)

if __name__ == '__main__':
    image = torch.randn(1, 1, 512, 512)  # 输入 xh 的形状为 [B C H/2 W2]
    channel_adapt_in = Channel_adapt(in_channels=1, out_channels=512)
    i0 = channel_adapt_in(image)
    print(i0.shape)

    image = torch.randn(16, 1, 64, 64)  # 输入 xh 的形状为 [B C H/2 W2]
    adapt_in = Verse_adapt(in_channels=1, out_channels=16)
    i1 = adapt_in(image)
    print(i1.shape)
