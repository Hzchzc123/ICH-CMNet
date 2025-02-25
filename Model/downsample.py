import torch
import torch.nn as nn

class DownsampleNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleNetwork, self).__init__()
        # 使用卷积层来进行下采样
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=2, padding=1)  # (batch_size, 128, 256, 256)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)  # (batch_size, 64, 128, 128)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=2, padding=1)  # (batch_size, out_channels, 64, 64)

    def forward(self, x):
        # 按顺序进行卷积操作，逐步下采样
        x1 = self.conv1(x)  # 下采样，输出大小减半
        x2 = self.conv2(x1)  # 再次下采样
        x3 = self.conv3(x2)  # 最后一次下采样
        return x1, x2, x3

if __name__ == '__main__':
    # 假设输入是(batch_size, channels, height, width)，注意 PyTorch 中的形状是 (batch_size, channels, height, width)
    input_tensor = torch.randn(16, 512, 512, 512)  # 模拟1张 512x512x512 的图片

    # 创建模型
    downsample_network = DownsampleNetwork(in_channels=512, out_channels=16)

    # 前向传播
    output1, output2, output3 = downsample_network(input_tensor)

    # 输出的形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape 1: {output1.shape}")
    print(f"Output shape 2: {output2.shape}")
    print(f"Output shape 3: {output3.shape}")
