import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN_UpSampling(nn.Module):
    def __init__(self, channels_list):
        """
        初始化FPN上采样模块，支持4层特征图的上采样。

        :param in_channels_list: 每一层输入特征图的通道数 [C3, C4, C5, C6]
        :param out_channels: 最终输出的特征图通道数（通常是256）
        """
        super(FPN_UpSampling, self).__init__()

        in_channels_list = channels_list[:-1]
        out_channels_list = channels_list[1:]

        # 用于上采样的卷积层，逐步上采样
        self.upsample_convs = nn.ModuleList([
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2,padding=1)
            for in_channels,out_channels in zip(in_channels_list,out_channels_list)  # 只需要从C6到C5, C5到C4, C4到C3上采样
        ])

    def forward(self, c3, c4, c5):
        """
        FPN的上采样过程，依次进行横向连接与上采样。

        :param c3: 从骨干网络提取的C3特征图
        :param c4: 从骨干网络提取的C4特征图
        :param c5: 从骨干网络提取的C5特征图
        :param c6: 从骨干网络提取的C6特征图
        :return: 上采样后融合的多尺度特征图
        """
        # 横向连接（Lateral connections）
        p5 = c5
        p4 = c4+self.upsample_convs[0](p5)  # C5
        p3 = c3+self.upsample_convs[1](p4)  # C4
        p2 = self.upsample_convs[2](p3)  # C4
        # 返回多尺度特征图
        return p2


# 测试FPN上采样
if __name__ == '__main__':
    # 假设C3, C4, C5, C6是骨干网络提取的特征图，通道数分别为256, 512, 1024, 2048
    c3 = torch.randn(1, 256, 256, 256)
    c4 = torch.randn(1, 128, 128, 128)
    c5 = torch.randn(1, 32, 64, 64)

    fpn_upsample = FPN_UpSampling([32, 128, 256, 1])
    outputs = fpn_upsample(c3, c4, c5)

    # 输出每个特征图的形状
    print(outputs.shape)

