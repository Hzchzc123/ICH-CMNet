from torch import nn
import torch
from Model.downsample import DownsampleNetwork
from Model.channel_adapt import Channel_adapt
from Model.CLIP_train.clip import CLIP
from Model.Mamba_downsample import MambaLayer
import torch.nn.functional as F  # 用于Softmax

class feature_clip(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.channel_adapt = Channel_adapt(in_channels=1, out_channels=64)  # (16,64,512,512)
        self.downsample = DownsampleNetwork(in_channels=64, out_channels=16)  # (16,16,64,64)
        self.downsample_clip_adapt = Channel_adapt(in_channels=16, out_channels=1)  # (16,1,64,64)
        self.clip = CLIP()

    def forward(self, img_x, text_x):
        channel_x = self.channel_adapt(img_x)  # (16,64,512,512)
        _, _, downsample_x = self.downsample(channel_x)  # (16,16,64,64)
        adapt_downsample = self.downsample_clip_adapt(downsample_x)
        clip_x = self.clip(adapt_downsample, text_x)

        # 添加 Softmax 在最后一层
        clip_x = F.softmax(clip_x, dim=1)  # softmax 作用在类维度上（dim=1）

        return clip_x


# 测试代码
if __name__ == '__main__':
    img_x = torch.randn(16, 1, 512, 512)  # 批量大小为32，3通道，512x512的图像
    text_x = torch.randint(0, 16, (16, 1))

    # 使用 torch.device 来转换为设备对象
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = feature_clip(16).to(DEVICE)
    output = model(img_x.to(DEVICE), text_x.to(DEVICE))
    print(output)
    print(output.shape)
