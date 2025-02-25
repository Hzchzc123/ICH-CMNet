from encodings.punycode import adapt
from torch import nn
import torch
from .img_encoder import ImgEncoder
from .text_encoder import TextEncoder
from Model.channel_adapt import Channel_adapt
class CLIP(nn.Module):
    def __init__(self,):
        super().__init__()
        self.img_enc=ImgEncoder(1)
        self.text_enc=TextEncoder()

    def forward(self,img_x,text_x):
        img_emb=self.img_enc(img_x)
        text_emb=self.text_enc(text_x)
        return img_emb@text_emb.T

if __name__=='__main__':
    clip=CLIP()
    img_x=torch.randn(32,16,64,64)
    text_x=torch.randint(0,16,(32,1))
    print(img_x.shape)
    print(text_x.shape)
    logits=clip(img_x,text_x)
    print(logits.shape)
