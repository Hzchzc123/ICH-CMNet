import torch.nn as nn
import torch.nn.functional as F
from Model.Mamba_downsample import MambaLayer
from Model.channel_adapt import Channel_adapt,Verse_Channel_adapt,Verse_adapt
from Model.downsample import DownsampleNetwork
from Model.SAM.SAM import Sam
from Model.SAM.image_encoder import ImageEncoderViT
from Model.SAM.mask_decoder import MaskDecoder
from Model.SAM.prompt_encoder import PromptEncoder
from Model.CLIP_train.utils import clip_fusion
import torch
from Model.FPN_upsample import FPN_UpSampling
from Model.gab import group_aggregation_bridge

class Net(nn.Module):
    def __init__(self,):
        super().__init__()

        self.channel_adapt_in = Channel_adapt(in_channels=1,out_channels=256)

        self.Down = DownsampleNetwork(in_channels=256,out_channels=16)

        self.Mamba1 = MambaLayer(in_chs=128, dim=32, d_state=16, d_conv=4, expand=2, last_feat_size=16)
        self.Mamba2 = MambaLayer(in_chs=64, dim=16, d_state=16, d_conv=4, expand=2, last_feat_size=16)
        self.Mamba3 = MambaLayer(in_chs=16, dim=4, d_state=16, d_conv=4, expand=2, last_feat_size=16)

        self.downsample_clip_adapt3 = Channel_adapt(in_channels=32, out_channels=1)  # (16,1,64,64)

        self.clip_upsample3 = Verse_Channel_adapt(in_channels=1, out_channels=32)#(16,32,64,64)

        self.clip_upsample2 = Verse_adapt(in_channels=32, out_channels=128)#(16,128,128,128)

        self.clip_upsample1 = Verse_adapt(in_channels=128, out_channels=256)#(16,256,256,256)

        self.sam1 = Sam(image_encoder=(ImageEncoderViT(img_size=64,in_chans=256)),prompt_encoder=PromptEncoder,mask_decoder=MaskDecoder)
        self.sam2 = Sam(image_encoder=(ImageEncoderViT(img_size=64, in_chans=128)),prompt_encoder=PromptEncoder,mask_decoder=MaskDecoder)
        self.sam3 = Sam(image_encoder=(ImageEncoderViT(img_size=64, in_chans=32)),prompt_encoder=PromptEncoder,mask_decoder=MaskDecoder)

        self.gab3 = group_aggregation_bridge(dim_xh=32, dim_xl=32)
        self.gab2 = group_aggregation_bridge(dim_xh=32, dim_xl=128)
        self.gab1 = group_aggregation_bridge(dim_xh=128, dim_xl=256)

        self.Up_sample = FPN_UpSampling([32, 128, 256, 1])


    def forward(self, image, mask):

        image0 = image

        i0 = self.channel_adapt_in(image0)

        i1,i2,i3 = self.Down(i0) #(16,128,256,256),(16,64,128,128),(16,16,64,64)

        mamba1 = self.Mamba1(i1)#(16,256,256,256)
        mamba2 = self.Mamba2(i2)#(16,128,128,128)
        mamba3 = self.Mamba3(i3)#(16,32,64,64)
        
        adapt_downsample3 = self.downsample_clip_adapt3(mamba3)  # (16,1,64,64)
        clip_fusion3 = clip_fusion(adapt_downsample3)
        clip_upsample3 = self.clip_upsample3(clip_fusion3)  # (16,32,64,64)

        clip_upsample2 = self.clip_upsample2(clip_upsample3)#(16,128,128,128)

        clip_upsample1 = self.clip_upsample1(clip_upsample2)#(16,256,256,256)
        
        batch_size1 = clip_upsample1.size(0)
        masks_list1 = []
        for i in range(batch_size1):
            sam1 = self.sam1([{'image': clip_upsample1[i, :, :, :], 'mask_inputs': mask[i, :, :, :],
                               'original_size': (512, 512), }], multimask_output=False)
            sam_mask1 = sam1[0]['masks']
            masks_list1.append(sam_mask1)
        # 使用 torch.cat 合并张量
        sam_stack1 = torch.cat(masks_list1, dim=0)  # 在第 0 维连接，结果是 (16, 1, 512, 512)

        batch_size2 = clip_upsample2.size(0)
        masks_list2 = []
        for i in range(batch_size2):
            sam2 = self.sam2([{'image': clip_upsample2[i, :, :, :], 'mask_inputs': mask[i, :, :, :],
                               'original_size': (512, 512), }], multimask_output=False)
            sam_mask2 = sam2[0]['masks']
            masks_list2.append(sam_mask2)
        # 使用 torch.cat 合并张量
        sam_stack2 = torch.cat(masks_list2, dim=0)  # 在第 0 维连接，结果是 (16, 1, 512, 512)

        batch_size3 = clip_upsample3.size(0)
        masks_list3 = []
        for i in range(batch_size3):
            sam3 = self.sam3([{'image': clip_upsample3[i, :, :, :], 'mask_inputs': mask[i, :, :, :],
                               'original_size': (512, 512), }], multimask_output=False)
            sam_mask3 = sam3[0]['masks']
            masks_list3.append(sam_mask3)
        # 使用 torch.cat 合并张量
        sam_stack3 = torch.cat(masks_list3, dim=0)  # 在第 0 维连接，结果是 (16, 1, 512, 512)

        gab_output3 = self.gab3(xh=mamba3,xl=mamba3,mask=sam_stack3)#(16,32,64,64)
        gab_output2 = self.gab2(xh=gab_output3,xl=mamba2,mask=sam_stack2)#(16,128,128,128)
        gab_output1 = self.gab1(xh=gab_output2,xl=mamba1,mask=sam_stack1)#(16,256,256,256)

        out_put = self.Up_sample(gab_output1,gab_output2,gab_output3)

        return out_put

if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    image = torch.randn(2, 1, 512, 512).to(DEVICE)  # 输入 xh 的形状为 [B C H/2 W2]
    model = Net().to(DEVICE)
    mask = torch.randn(2, 1, 512, 512).to(DEVICE)
    output = model(image,mask)
    print(output.shape)
